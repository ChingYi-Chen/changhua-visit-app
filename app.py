# app.py
import os
import re
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import pandas as pd
import streamlit as st
import googlemaps
import folium
from streamlit_folium import st_folium
import bcrypt

from auth import hash_password
from db import get_conn, init_db  # single source of truth for DB connection + schema

# =====================
# 基本設定
# =====================
st.set_page_config(page_title="彰化訪視排程系統", layout="wide")

ORIGIN_ADDRESS = "彰化縣政府第二辦公大樓"
SUBSIDY_PER_KM = 3.0
MAX_WAYPOINTS_FOR_DIRECTIONS = 10  # Google Directions 的 waypoint 實務上也有上限；此處保守

GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()

# 管理員（只用於「第一次自動建立管理員帳號」以及判斷是否顯示管理頁）
INIT_ADMIN_USER = os.getenv("INIT_ADMIN_USER", "").strip()
INIT_ADMIN_PASS = os.getenv("INIT_ADMIN_PASS", "").strip()

# =====================
# Helpers
# =====================
def case_sort_key(case_id: str):
    """Natural sort for IDs like B001, A12, B2 (B2 < B10)."""
    s = str(case_id).strip()
    m = re.match(r"^([A-Za-z]+)\s*0*(\d+)", s)
    if m:
        return (m.group(1).upper(), int(m.group(2)))
    return ("ZZZ", 10**18, s)

def build_case_label_maps(df: pd.DataFrame):
    """Return (labels_sorted, label->case_id, case_id->label)."""
    if df is None or df.empty:
        return [], {}, {}
    cols = [c for c in ["case_id", "name"] if c in df.columns]
    tmp = df[cols].copy()
    if "case_id" not in tmp.columns:
        return [], {}, {}
    tmp["case_id"] = tmp["case_id"].astype(str).str.strip()
    if "name" in tmp.columns:
        tmp["name"] = tmp["name"].fillna("").astype(str).str.strip()
    else:
        tmp["name"] = ""
    tmp = tmp[tmp["case_id"].astype(str).str.strip() != ""]
    tmp = tmp.drop_duplicates(subset=["case_id"], keep="last")
    tmp = tmp.sort_values(by="case_id", key=lambda s: s.map(case_sort_key))

    def _mk_label(r):
        nm = r.get("name", "")
        return f"{r['case_id']}｜{nm}" if nm else f"{r['case_id']}"

    tmp["label"] = tmp.apply(_mk_label, axis=1)
    labels = tmp["label"].tolist()
    label_to_id = dict(zip(tmp["label"], tmp["case_id"]))
    id_to_label = dict(zip(tmp["case_id"], tmp["label"]))
    return labels, label_to_id, id_to_label

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in meters (rough) used only for matching a clicked point to nearest marker."""
    import math
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d1 = math.radians(lat2 - lat1)
    d2 = math.radians(lon2 - lon1)
    a = math.sin(d1/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(d2/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# =====================
# Auth
# =====================
def verify_user(username: str, password: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    ok = bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8"))
    if not ok:
        return None
    return {"user_id": row["id"], "username": row["username"]}

def create_user(username: str, password: str) -> Tuple[bool, str]:
    """Create a user. Returns (ok, message)."""
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return False, "帳號與密碼不得為空"
    if len(password) < 6:
        return False, "密碼至少 6 碼"
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        return True, "已建立帳號"
    except Exception as e:
        # 常見：UNIQUE constraint failed
        msg = str(e)
        if "UNIQUE" in msg or "unique" in msg:
            return False, "此帳號已存在"
        return False, f"建立失敗：{msg}"
    finally:
        conn.close()

def init_admin_if_needed():
    """
    Render 上常見做法：用環境變數 INIT_ADMIN_USER/INIT_ADMIN_PASS 在「第一次部署」時自動建立管理員。
    - 若帳號已存在：不動作
    - 若未提供 env：不動作
    """
    if not INIT_ADMIN_USER or not INIT_ADMIN_PASS:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username = ?", (INIT_ADMIN_USER,))
    exists = cur.fetchone() is not None
    conn.close()
    if exists:
        return
    create_user(INIT_ADMIN_USER, INIT_ADMIN_PASS)

# =====================
# 工具：地址清洗
# =====================
def normalize_addr(addr: str) -> str:
    if addr is None:
        return ""
    s = str(addr).strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("臺", "台")
    return s

# =====================
# Google Geocoding（含快取表 geocode_cache）
# =====================
def cache_get(addr_norm: str) -> Optional[Tuple[float, float]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT lat, lng FROM geocode_cache WHERE addr_norm = ?", (addr_norm,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return float(row["lat"]), float(row["lng"])

def cache_set(addr_norm: str, lat: float, lng: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      INSERT OR REPLACE INTO geocode_cache (addr_norm, lat, lng, updated_at)
      VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """, (addr_norm, lat, lng))
    conn.commit()
    conn.close()

# =====================
# Distance cache (Driving distance in meters) for TSP optimization
# =====================
def _loc_key_from_latlng(lat: float, lng: float) -> str:
    # Round to avoid tiny floating diffs; ~0.11m at 6 decimals lat
    return f"{float(lat):.6f},{float(lng):.6f}"

def dist_cache_get(src: str, dst: str) -> Optional[int]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT meters FROM distance_cache WHERE src=? AND dst=?", (src, dst))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        return int(row["meters"])
    except Exception:
        return None

def dist_cache_set(src: str, dst: str, meters: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO distance_cache (src, dst, meters, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        (src, dst, int(meters)),
    )
    conn.commit()
    conn.close()

def init_distance_cache_table():
    """Ensure distance_cache exists. Safe to call on every start."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS distance_cache (
        src TEXT NOT NULL,
        dst TEXT NOT NULL,
        meters INTEGER NOT NULL,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (src, dst)
      )
    """)
    conn.commit()
    conn.close()

def get_distance_matrix_m_cached(gmaps: googlemaps.Client, origin_addr: str, points: List[Tuple[float, float]]):
    """
    Return (N+1)x(N+1) matrix in meters for: [origin] + points.
    Uses SQLite cache; if any pair missing, fetch full matrix once then cache all pairs.
    """
    locs = [origin_addr] + [f"{lat},{lng}" for lat, lng in points]
    keys = [f"ADDR:{normalize_addr(origin_addr)}"] + [_loc_key_from_latlng(lat, lng) for lat, lng in points]

    n = len(locs)
    # 1) Try build from cache
    mat = [[None]*n for _ in range(n)]
    missing = False
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0
                continue
            m = dist_cache_get(keys[i], keys[j])
            if m is None:
                missing = True
            else:
                mat[i][j] = m

    # 2) If any missing, fetch full matrix once, cache everything, then fill mat
    if missing:
        try:
            resp = gmaps.distance_matrix(
                origins=locs,
                destinations=locs,
                mode="driving",
            )
        except Exception:
            return None

        rows = resp.get("rows", []) if isinstance(resp, dict) else []
        if not rows or len(rows) != n:
            return None

        for i in range(n):
            elems = rows[i].get("elements", []) if isinstance(rows[i], dict) else []
            if len(elems) != n:
                return None
            for j in range(n):
                if i == j:
                    mat[i][j] = 0
                    continue
                el = elems[j]
                if not isinstance(el, dict) or el.get("status") != "OK":
                    # If any pair fails, keep it very large to avoid breaking TSP; also do not cache failure.
                    mat[i][j] = 10**12
                    continue
                meters = int(el.get("distance", {}).get("value", 0))
                mat[i][j] = meters
                dist_cache_set(keys[i], keys[j], meters)

    # ensure all filled
    for i in range(n):
        for j in range(n):
            if mat[i][j] is None:
                return None
    return mat



def geocode_address(gmaps: googlemaps.Client, address: str) -> Optional[Tuple[float, float]]:
    addr_norm = normalize_addr(address)
    if not addr_norm:
        return None

    cached = cache_get(addr_norm)
    if cached:
        return cached

    try:
        res = gmaps.geocode(address)
        if not res:
            return None
        loc = res[0]["geometry"]["location"]
        lat, lng = float(loc["lat"]), float(loc["lng"])
        cache_set(addr_norm, lat, lng)
        return lat, lng
    except Exception:
        return None

# =====================
# 讀取 Excel：自動找表頭列（跳過合併標題列）
# =====================
REQUIRED_KEYS = ["案號", "姓名", "現居地址"]

def find_header_row(df_raw: pd.DataFrame) -> Optional[int]:
    for i in range(min(len(df_raw), 60)):
        row = df_raw.iloc[i].astype(str).fillna("")
        joined = " ".join(row.tolist())
        hit = sum(1 for k in REQUIRED_KEYS if k in joined)
        if hit >= 2:
            return i
    return None

def read_excel_autodetect(file) -> pd.DataFrame:
    # 1) header=None 把整張表讀進來（可處理合併標題列）
    df_raw = pd.read_excel(file, header=None, dtype=object)

    # 2) 自動找表頭列（包含：案號/姓名/現居地址）
    hdr = find_header_row(df_raw)
    if hdr is None:
        raise ValueError("找不到表頭列（請確認 Excel 內包含：案號/姓名/現居地址）")

    headers = df_raw.iloc[hdr].astype(str).tolist()

    # 3) 取表頭列之後的資料，並套用欄名
    df_full = df_raw.iloc[hdr + 1:].copy()
    df_full.columns = headers
    df_full = df_full.dropna(how="all")
    df_full.columns = [re.sub(r"\s+", "", str(c)) for c in df_full.columns]

    # 4) 基本清洗
    if "案號" not in df_full.columns:
        raise ValueError(f"Excel 缺少必要欄位：['案號']\n目前欄位：{list(df_full.columns)}")

    df_full["案號"] = df_full["案號"].astype(str).str.strip()
    df_full = df_full[df_full["案號"].str.len() > 0]
    df_full = df_full[~df_full["案號"].str.contains("案號|nan|None", na=False)]

    missing = [c for c in REQUIRED_KEYS if c not in df_full.columns]
    if missing:
        raise ValueError(f"Excel 缺少必要欄位：{missing}\n目前欄位：{list(df_full.columns)}")

    keep_cols = ["案號", "姓名", "現居地址"]
    if "鄉鎮" in df_full.columns:
        keep_cols.append("鄉鎮")
    df = df_full[keep_cols].copy()
    df = df.rename(columns={"案號": "case_id", "姓名": "name", "現居地址": "address", "鄉鎮": "town"})

    # 已移除家訪日期/最後家訪日解析（依目前需求）
    return df

# =====================
# Cases CRUD
# =====================
def fetch_cases(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("""
      SELECT case_id, name, address_raw, address_fixed, town, lat, lng, geo_status, updated_at
      FROM cases
      WHERE user_id = ?
      ORDER BY updated_at DESC
    """, conn, params=(user_id,))
    conn.close()
    return df

def upsert_case(user_id: int, case_id: str, name: str, address: str, town: str,
                lat: Optional[float], lng: Optional[float],
                geo_status: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO cases (user_id, case_id, name, address_raw, town, lat, lng, geo_status, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
      ON CONFLICT(user_id, case_id) DO UPDATE SET
        name=excluded.name,
        address_raw=excluded.address_raw,
        town=excluded.town,
        lat=excluded.lat,
        lng=excluded.lng,
        geo_status=excluded.geo_status,
        updated_at=CURRENT_TIMESTAMP
    """, (user_id, case_id, name, address, town, lat, lng, geo_status))
    conn.commit()
    conn.close()

def update_case_address(user_id: int, case_id: str, address_fixed: str,
                        lat: Optional[float], lng: Optional[float], geo_status: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      UPDATE cases
      SET address_fixed=?, lat=?, lng=?, geo_status=?, updated_at=CURRENT_TIMESTAMP
      WHERE user_id=? AND case_id=?
    """, (address_fixed, lat, lng, geo_status, user_id, case_id))
    conn.commit()
    conn.close()

def update_case_latlng(user_id: int, case_id: str, lat: float, lng: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      UPDATE cases
      SET lat=?, lng=?, geo_status='MANUAL', updated_at=CURRENT_TIMESTAMP
      WHERE user_id=? AND case_id=?
    """, (lat, lng, user_id, case_id))
    conn.commit()
    conn.close()

def delete_case(user_id: int, case_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM cases WHERE user_id=? AND case_id=?", (user_id, case_id))
    conn.commit()
    conn.close()

# =====================
# 路徑最佳化（道路距離）：
# - 用 Distance Matrix 取「實際道路距離矩陣」
# - 用 Held–Karp DP 求「全域最短閉環」（起終點=ORIGIN）
# - 最後用 Directions 依最佳順序產出可導航連結與精準里程
# =====================
def get_distance_matrix_m(gmaps: googlemaps.Client, origin_addr: str, points: List[Tuple[float, float]]) -> List[List[int]]:
    """
    Return matrix (n+1)x(n+1) in meters.
    Index 0 = origin_addr, 1..n = points.
    """
    locs = [origin_addr] + [f"{lat},{lng}" for (lat, lng) in points]
    try:
        resp = gmaps.distance_matrix(origins=locs, destinations=locs, mode="driving")
    except Exception:
        return []

    rows = resp.get("rows", [])
    n = len(locs)
    M = [[10**15 for _ in range(n)] for _ in range(n)]
    for i in range(min(n, len(rows))):
        elems = rows[i].get("elements", [])
        for j in range(min(n, len(elems))):
            e = elems[j]
            if e.get("status") == "OK":
                M[i][j] = int(e.get("distance", {}).get("value", 10**15))
            elif i == j:
                M[i][j] = 0
    # ensure diagonal zero
    for i in range(n):
        M[i][i] = 0
    return M

def tsp_held_karp_cycle(dist: List[List[int]]) -> List[int]:
    """
    Solve TSP cycle starting/ending at 0 visiting all nodes 1..n-1 once.
    Returns visit order of nodes (excluding 0) in optimal sequence.
    If dist invalid, returns [].
    """
    if not dist or not dist[0]:
        return []
    n = len(dist)
    if n <= 2:
        return [1] if n == 2 else []

    # DP over subsets of {1..n-1}
    # dp[mask][i] = min cost to start at 0, visit mask, end at i (i in mask)
    size = 1 << (n - 1)
    INF = 10**15
    dp = [[INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]

    # init
    for i in range(1, n):
        m = 1 << (i - 1)
        dp[m][i] = dist[0][i]
        parent[m][i] = 0

    for mask in range(size):
        for last in range(1, n):
            if not (mask & (1 << (last - 1))):
                continue
            prev_mask = mask ^ (1 << (last - 1))
            if prev_mask == 0:
                continue
            # try prev
            best = dp[mask][last]
            best_prev = parent[mask][last]
            for prev in range(1, n):
                if not (prev_mask & (1 << (prev - 1))):
                    continue
                cand = dp[prev_mask][prev] + dist[prev][last]
                if cand < best:
                    best = cand
                    best_prev = prev
            dp[mask][last] = best
            parent[mask][last] = best_prev

    full = size - 1
    # close the tour
    best_cost = INF
    best_last = -1
    for last in range(1, n):
        cand = dp[full][last] + dist[last][0]
        if cand < best_cost:
            best_cost = cand
            best_last = last

    if best_last == -1:
        return []

    # reconstruct path
    order_rev = []
    mask = full
    last = best_last
    while last != 0 and last != -1:
        order_rev.append(last)
        prev = parent[mask][last]
        mask = mask ^ (1 << (last - 1))
        last = prev

    order = list(reversed(order_rev))
    # return indices 1..n-1
    return order

def calc_route_shortest(gmaps: googlemaps.Client, origin_addr: str, points: List[Tuple[float, float]]):
    """
    Return (ordered_points_indices, total_meters_from_directions, url)
    ordered_points_indices: indices in original points list (0..len(points)-1) in visit order.
    """
    if not points:
        return [], 0, ""

    if len(points) == 1:
        # trivial: origin -> p -> origin
        ordered_points = points
        directions = gmaps.directions(origin=origin_addr, destination=origin_addr, mode="driving",
                                      waypoints=[f"{points[0][0]},{points[0][1]}"], alternatives=False)
        total_m = 0
        if directions:
            for leg in directions[0].get("legs", []):
                total_m += int(leg.get("distance", {}).get("value", 0))
        url = build_gmaps_dir_url(origin_addr, origin_addr, [f"{points[0][0]},{points[0][1]}"])
        return [0], total_m, url

    # 1) Distance Matrix
    dist = get_distance_matrix_m_cached(gmaps, origin_addr, points)
    if not dist:
        # fallback: use Directions optimize (heuristic)
        return calc_route_heuristic(gmaps, origin_addr, points)

    # 2) Solve optimal order (node indices in [1..n])
    node_order = tsp_held_karp_cycle(dist)  # nodes including origin index 0
    if not node_order:
        return calc_route_heuristic(gmaps, origin_addr, points)

    # map node indices -> points indices
    ordered_point_idx = [i - 1 for i in node_order]  # node 1 => points[0]
    ordered_waypoints = [f"{points[i][0]},{points[i][1]}" for i in ordered_point_idx]

    # 3) Call Directions with fixed order (no optimize:true)
    directions = gmaps.directions(
        origin=origin_addr,
        destination=origin_addr,
        mode="driving",
        waypoints=ordered_waypoints,
        alternatives=False
    )
    total_m = 0
    if directions:
        for leg in directions[0].get("legs", []):
            total_m += int(leg.get("distance", {}).get("value", 0))

    url = build_gmaps_dir_url(origin_addr, origin_addr, ordered_waypoints)
    return ordered_point_idx, total_m, url

def build_gmaps_dir_url(origin: str, destination: str, waypoints: List[str]) -> str:
    from urllib.parse import quote
    origin_q = quote(origin)
    dest_q = quote(destination)
    wp_q = "|".join([quote(w) for w in waypoints])
    return f"https://www.google.com/maps/dir/?api=1&origin={origin_q}&destination={dest_q}&waypoints={wp_q}&travelmode=driving"

def calc_route_heuristic(gmaps: googlemaps.Client, origin_addr: str, points: List[Tuple[float, float]]):
    """Fallback to Google Directions optimize:true (heuristic)."""
    waypoints = [f"{lat},{lng}" for lat, lng in points]
    if len(waypoints) > MAX_WAYPOINTS_FOR_DIRECTIONS:
        waypoints = waypoints[:MAX_WAYPOINTS_FOR_DIRECTIONS]

    wp_param = ["optimize:true"] + waypoints
    directions = gmaps.directions(
        origin=origin_addr,
        destination=origin_addr,
        mode="driving",
        waypoints=wp_param,
        alternatives=False
    )
    if not directions:
        return [], 0, ""

    route = directions[0]
    order = route.get("waypoint_order", [])
    total_m = 0
    for leg in route.get("legs", []):
        total_m += int(leg.get("distance", {}).get("value", 0))

    ordered_waypoints = waypoints
    if order and len(order) == len(waypoints):
        ordered_waypoints = [waypoints[i] for i in order]
    url = build_gmaps_dir_url(origin_addr, origin_addr, ordered_waypoints)

    # order here is index in 'waypoints', map back to points indices
    if order and len(order) == len(waypoints):
        return order, total_m, url
    return list(range(len(waypoints))), total_m, url

# =====================
# UI
# =====================
def login_view():
    st.title("彰化訪視排程系統｜登入")
    u = st.text_input("帳號")
    p = st.text_input("密碼", type="password")
    if st.button("登入"):
        user = verify_user(u.strip(), p)
        if not user:
            st.error("帳號或密碼錯誤")
        else:
            st.session_state["user"] = user
            st.session_state.setdefault("selected_case_ids", [])
            st.session_state.setdefault("picked_labels", [])
            st.success("登入成功")
            st.rerun()

def page_admin_create_users(user):
    st.header("管理員｜建立使用者帳號")
    st.caption("此頁面僅管理員可見，用於替教授/學長建立帳號。")

    with st.form("create_user_form", clear_on_submit=True):
        new_u = st.text_input("新帳號")
        new_p = st.text_input("新密碼（至少 6 碼）", type="password")
        new_p2 = st.text_input("再次輸入密碼", type="password")
        submitted = st.form_submit_button("建立帳號")
    if submitted:
        if new_p != new_p2:
            st.error("兩次密碼不一致")
        else:
            ok, msg = create_user(new_u, new_p)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

def page_import_excel(user):
    st.header("Excel 匯入（案號為主鍵，重複則更新）")

    if not GOOGLE_KEY:
        st.error("尚未設定 GOOGLE_MAPS_API_KEY。請先設定環境變數後再使用定位/路線功能。")
        st.stop()

    uploaded = st.file_uploader("上傳 Excel（含案號/姓名/現居地址）", type=["xlsx"])
    if not uploaded:
        st.info("請上傳檔案。")
        return

    gmaps = googlemaps.Client(key=GOOGLE_KEY)

    with st.spinner("正在讀取並自動偵測表頭列..."):
        df = read_excel_autodetect(uploaded)

    st.success(f"讀取完成：{len(df)} 筆")
    st.dataframe(df.head(20), use_container_width=True)

    do_geocode = st.checkbox("匯入時自動定位（建議勾選）", value=True)

    if st.button("開始匯入到我的地圖資料"):
        ok, fail = 0, 0
        with st.spinner("匯入中..."):
            for _, r in df.iterrows():
                case_id = str(r.get("case_id", "")).strip()
                name = str(r.get("name", "")).strip()
                addr = str(r.get("address", "")).strip()
                town = str(r.get("town", "")).strip() if "town" in df.columns else ""
                lat = lng = None
                status = "FAIL"
                if do_geocode and addr:
                    geo = geocode_address(gmaps, addr)
                    if geo:
                        lat, lng = geo
                        status = "OK"

                upsert_case(
                    user_id=user["user_id"],
                    case_id=case_id,
                    name=name,
                    address=addr,
                    town=town,
                    lat=lat,
                    lng=lng,
                    geo_status=status
                )

                if status == "OK":
                    ok += 1
                else:
                    fail += 1

        st.success(f"匯入完成：OK={ok}｜FAIL={fail}")
        st.info("若 FAIL，請到『個案管理』用「改地址」或「地圖點落點』修正。")
        st.rerun()

def page_manage_cases(user):
    st.header("個案管理（新增 / 刪除 / 手動修正）")
    df = fetch_cases(user["user_id"])

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("新增個案")
        with st.form("add_case_form", clear_on_submit=True):
            case_id = st.text_input("案號")
            name = st.text_input("姓名")
            addr = st.text_input("地址")
            submitted = st.form_submit_button("新增並定位")
        if submitted:
            if not case_id.strip() or not name.strip():
                st.error("案號與姓名必填")
            else:
                lat = lng = None
                status = "FAIL"
                if GOOGLE_KEY and addr.strip():
                    gmaps = googlemaps.Client(key=GOOGLE_KEY)
                    geo = geocode_address(gmaps, addr.strip())
                    if geo:
                        lat, lng = geo
                        status = "OK"

                upsert_case(
                    user_id=user["user_id"],
                    case_id=case_id.strip(),
                    name=name.strip(),
                    address=addr.strip(),
                    town="",
                    lat=lat,
                    lng=lng,
                    geo_status=status,
                )
                st.success("新增完成")
                st.rerun()

    with c2:
        st.subheader("刪除個案")
        del_labels, del_label_to_id, _ = build_case_label_maps(df)
        del_label = st.selectbox("選擇個案", options=[""] + del_labels, key="del_case_label")
        del_id = del_label_to_id.get(del_label, "") if del_label else ""
        if st.button("刪除選取案號"):
            if del_id:
                delete_case(user["user_id"], del_id)
                st.success("已刪除")
                st.rerun()

    st.divider()

    st.subheader("修正方式 1：改地址並重新定位")
    fix_labels, fix_label_to_id, _ = build_case_label_maps(df)
    fix_label = st.selectbox("選擇要修正的個案", options=[""] + fix_labels, key="fix_addr_label")
    fix_id = fix_label_to_id.get(fix_label, "") if fix_label else ""
    if fix_id:
        row = df[df["case_id"] == fix_id].iloc[0]
        current_addr = row["address_fixed"] or row["address_raw"] or ""
        new_addr = st.text_input("修正地址", value=str(current_addr), key="fix_addr_text")

        if st.button("套用地址修正並重新定位"):
            if not GOOGLE_KEY:
                st.error("未設定 GOOGLE_MAPS_API_KEY")
            else:
                gmaps = googlemaps.Client(key=GOOGLE_KEY)
                geo = geocode_address(gmaps, new_addr)
                if geo:
                    lat, lng = geo
                    update_case_address(user["user_id"], fix_id, new_addr, lat, lng, "OK")
                    st.success("修正完成：已重新定位")
                else:
                    update_case_address(user["user_id"], fix_id, new_addr, None, None, "FAIL")
                    st.warning("修正已保存，但仍無法定位。建議用『地圖點落點』。")
                st.rerun()

    st.divider()

    st.subheader("修正方式 2：地圖點落點（最穩）")
    st.caption("流程：先選案號 → 在地圖上點一下 → 按『套用落點』")
    pin_labels, pin_label_to_id, _ = build_case_label_maps(df)
    pin_label = st.selectbox("選擇要套用落點的個案", options=[""] + pin_labels, key="pin_label")
    pin_id = pin_label_to_id.get(pin_label, "") if pin_label else ""

    m = folium.Map(location=[24.07, 120.54], zoom_start=11)
    for _, r in df.dropna(subset=["lat", "lng"]).iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lng"]],
            radius=5,
            tooltip=f"{r['case_id']}｜{r['name']}",
        ).add_to(m)

    out = st_folium(m, use_container_width=True, height=520)
    clicked = out.get("last_clicked") or out.get("last_object_clicked")

    if clicked:
        st.info(f"你點的位置：lat={clicked['lat']:.6f}, lng={clicked['lng']:.6f}")

    if st.button("套用落點到選定案號"):
        if not pin_id:
            st.error("請先選案號")
        elif not clicked:
            st.error("請先在地圖上點一下落點")
        else:
            update_case_latlng(user["user_id"], pin_id, float(clicked["lat"]), float(clicked["lng"]))
            st.success("已套用落點（MANUAL）")
            st.rerun()

    st.divider()
    st.subheader("資料總覽")
    st.dataframe(df, use_container_width=True)

def page_map_and_route(user):
    st.header("地圖與路線（選取個案 → 計算最短路線與里程補助）")

    if not GOOGLE_KEY:
        st.error("尚未設定 GOOGLE_MAPS_API_KEY。")
        st.stop()

    df = fetch_cases(user["user_id"])
    df_ok = df.dropna(subset=["lat", "lng"]).copy()
    if df_ok.empty:
        st.info("目前沒有已定位的個案。請先匯入或定位。")
        return

    # ---- selection state (fix #3: avoid "短暫重整後要再點一次") ----
    st.session_state.setdefault("selected_case_ids", [])
    st.session_state.setdefault("picked_labels", [])

    selected_ids = list(st.session_state["selected_case_ids"])

    # ---- Map (fix #2: click marker to toggle selection) ----
    m = folium.Map(location=[24.07, 120.54], zoom_start=11)
    for _, r in df_ok.iterrows():
        label = f"{r['case_id']}｜{r['name']}"
        is_sel = str(r["case_id"]) in set(selected_ids)
        color = "green" if is_sel else "blue"
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lng"])],
            radius=6,
            tooltip=label,
            popup=f"{label}<br><br>{(r['address_fixed'] or r['address_raw'] or '')}",
            color=color,
            fill=True,
            fill_opacity=0.8
        ).add_to(m)

    c1, c2 = st.columns([2, 1])
    with c1:
        out = st_folium(m, use_container_width=True, height=650)
        clicked = out.get("last_clicked") or out.get("last_object_clicked")
        if clicked:
            # nearest case within 80m -> toggle
            latc, lngc = float(clicked["lat"]), float(clicked["lng"])
            tmp = df_ok.copy()
            tmp["__d"] = tmp.apply(lambda r: haversine_m(latc, lngc, float(r["lat"]), float(r["lng"])), axis=1)
            nearest = tmp.sort_values("__d").head(1)
            if not nearest.empty and float(nearest.iloc[0]["__d"]) < 80:
                cid = str(nearest.iloc[0]["case_id"])
                cur = set(st.session_state["selected_case_ids"])
                if cid in cur:
                    cur.remove(cid)
                else:
                    cur.add(cid)
                # keep stable, sorted by case id (not by click order)
                st.session_state["selected_case_ids"] = sorted(list(cur), key=case_sort_key)
                st.rerun()

    with c2:
        st.subheader("選取個案")
        st.caption(f"起點/終點：{ORIGIN_ADDRESS}")
        st.caption(f"補助：每公里 {SUBSIDY_PER_KM} 元（Google 道路里程）")
        st.caption("可用右側清單勾選，也可直接點地圖上的標記加入/移除。")

        labels, label_to_id, id_to_label = build_case_label_maps(df_ok)

        # sync picked_labels from selected_case_ids only when picked_labels is empty (avoid overwriting user's UI state)
        if not st.session_state["picked_labels"]:
            st.session_state["picked_labels"] = [id_to_label[cid] for cid in st.session_state["selected_case_ids"] if cid in id_to_label]

        picked_labels = st.multiselect(
            "已定位個案列表（案號｜姓名）",
            options=labels,
            key="picked_labels",
        )
        picked_ids = [label_to_id[x] for x in picked_labels]
        picked_ids = sorted(list(dict.fromkeys(picked_ids)), key=case_sort_key)  # de-dup + stable
        st.session_state["selected_case_ids"] = picked_ids

        picked_df = df_ok[df_ok["case_id"].isin(picked_ids)].copy()
        if not picked_df.empty:
            st.dataframe(picked_df[["case_id", "name", "geo_status"]], use_container_width=True)

        st.divider()

        if st.button("🚗 計算最短路線（道路距離最佳化）"):
            if len(picked_ids) < 1:
                st.error("請至少選 1 個個案")
                st.stop()

            if len(picked_ids) > MAX_WAYPOINTS_FOR_DIRECTIONS:
                st.warning(f"你選了 {len(picked_ids)} 個點，先以前 {MAX_WAYPOINTS_FOR_DIRECTIONS} 個計算。")
                picked_ids = picked_ids[:MAX_WAYPOINTS_FOR_DIRECTIONS]
                picked_df = df_ok[df_ok["case_id"].isin(picked_ids)].copy()

            gmaps = googlemaps.Client(key=GOOGLE_KEY)

            # Ensure deterministic order of points list (so returned order indices map correctly)
            picked_df = picked_df.sort_values("case_id", key=lambda s: s.map(case_sort_key)).reset_index(drop=True)
            points = list(zip(picked_df["lat"].astype(float), picked_df["lng"].astype(float)))

            order_idx, total_m, url = calc_route_shortest(gmaps, ORIGIN_ADDRESS, points)

            total_km = total_m / 1000.0
            subsidy = total_km * SUBSIDY_PER_KM

            st.success("計算完成（以道路距離矩陣求全域最短閉環；最後用 Directions 產出可導航路線）")
            st.metric("道路總里程（km）", f"{total_km:.2f}")
            st.metric("里程補助（元）", f"{subsidy:.0f}")

            if order_idx and len(order_idx) == len(picked_df):
                ordered_df = picked_df.iloc[order_idx].copy()
            else:
                ordered_df = picked_df.copy()

            st.subheader("建議拜訪順序")
            st.dataframe(ordered_df[["case_id", "name"]], use_container_width=True)

            if url:
                st.markdown(f"### [🔗 開啟 Google Maps 導航]({url})")

def main():
    if "user" not in st.session_state:
        login_view()
        return

    user = st.session_state["user"]

    with st.sidebar:
        st.write(f"登入者：**{user['username']}**")
        pages = ["地圖與路線", "Excel 匯入", "個案管理"]
        # 只有管理員看得到「新增帳號」
        if INIT_ADMIN_USER and user["username"] == INIT_ADMIN_USER:
            pages.append("管理員｜新增帳號")
        page = st.radio("功能選單", pages, index=0)

        if st.button("登出"):
            st.session_state.pop("user", None)
            st.session_state.pop("selected_case_ids", None)
            st.session_state.pop("picked_labels", None)
            st.rerun()

        st.divider()
        st.caption("Google API Key 以環境變數 GOOGLE_MAPS_API_KEY 設定。")

    if page == "Excel 匯入":
        page_import_excel(user)
    elif page == "個案管理":
        page_manage_cases(user)
    elif page == "管理員｜新增帳號":
        page_admin_create_users(user)
    else:
        page_map_and_route(user)

# ---- Boot ----
init_db()
init_distance_cache_table()
init_admin_if_needed()
main()