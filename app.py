import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st
import googlemaps
import folium
from streamlit_folium import st_folium
import bcrypt

# =====================
# 基本設定
# =====================
st.set_page_config(page_title="彰化訪視排程系統", layout="wide")

ORIGIN_ADDRESS = "彰化縣政府第二辦公大樓"
SUBSIDY_PER_KM = 3.0
MAX_WAYPOINTS_FOR_DIRECTIONS = 10
DB_PATH = Path("local.db")

GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()

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

# =====================
# DB
# =====================
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cases (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      case_id TEXT NOT NULL,
      name TEXT NOT NULL,
      address_raw TEXT,
      address_fixed TEXT,
      town TEXT,
      lat REAL,
      lng REAL,
      geo_status TEXT DEFAULT 'FAIL',
      last_visit TEXT,
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(user_id, case_id),
      FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS geocode_cache (
      addr_norm TEXT PRIMARY KEY,
      lat REAL,
      lng REAL,
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()

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

# =====================
# 工具：地址清洗 / 家訪日期解析（同一格多日期取最後一次）
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
# Google Geocoding（含快取）
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
    # 1) 先用 header=None 把整張表讀進來（可處理合併標題列）
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

    # 4) 基本清洗：案號必須存在且不能是標題殘留
    if "案號" not in df_full.columns:
        raise ValueError(f"Excel 缺少必要欄位：['案號']\n目前欄位：{list(df_full.columns)}")

    df_full["案號"] = df_full["案號"].astype(str).str.strip()
    df_full = df_full[df_full["案號"].str.len() > 0]
    df_full = df_full[~df_full["案號"].str.contains("案號|nan|None", na=False)]

    missing = [c for c in REQUIRED_KEYS if c not in df_full.columns]
    if missing:
        raise ValueError(f"Excel 缺少必要欄位：{missing}\n目前欄位：{list(df_full.columns)}")

    # 5) 只抽出系統需要的欄位（保留家訪日期欄名，如果它存在）
    keep_cols = ["案號", "姓名", "現居地址"]
    if "鄉鎮" in df_full.columns:
        keep_cols.append("鄉鎮")
    df = df_full[keep_cols].copy()
    df = df.rename(columns={"案號": "case_id", "姓名": "name", "現居地址": "address", "鄉鎮": "town"})
    # 6) 已移除家訪日期/最後家訪日解析（依需求）

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
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
# Directions：道路里程 + optimize:true
# =====================
def calc_route(gmaps: googlemaps.Client, origin_addr: str, points: List[Tuple[float, float]]):
    if not points:
        return [], 0, ""

    origin = origin_addr
    destination = origin_addr

    waypoints = [f"{lat},{lng}" for lat, lng in points]
    if len(waypoints) > MAX_WAYPOINTS_FOR_DIRECTIONS:
        waypoints = waypoints[:MAX_WAYPOINTS_FOR_DIRECTIONS]

    wp_param = ["optimize:true"] + waypoints
    directions = gmaps.directions(
        origin=origin,
        destination=destination,
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

    from urllib.parse import quote

    # 用 Google 回傳的 waypoint_order 重新排列（讓你點出去的導航順序也一致）
    ordered_waypoints = waypoints
    if order and len(order) == len(waypoints):
        ordered_waypoints = [waypoints[i] for i in order]

    origin_q = quote(origin)
    dest_q = quote(destination)
    wp_q = "|".join([quote(w) for w in ordered_waypoints])
    url = f"https://www.google.com/maps/dir/?api=1&origin={origin_q}&destination={dest_q}&waypoints={wp_q}&travelmode=driving"

    return order, total_m, url

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
            st.success("登入成功")
            st.rerun()

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
        st.info("若 FAIL，請到『個案管理』用「改地址」或「地圖點落點」修正。")
        st.rerun()

def page_manage_cases(user):
    st.header("個案管理（新增 / 刪除 / 手動修正）")

    df = fetch_cases(user["user_id"])

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("新增個案")
        case_id = st.text_input("案號", key="new_case_id")
        name = st.text_input("姓名", key="new_name")
        addr = st.text_input("地址", key="new_addr")
        town = ""  # 移除輸入：鄉鎮
        if st.button("新增並定位"):
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
                    town=town.strip(),
                    lat=lat,
                    lng=lng,
                    geo_status=status,                )
                st.success("新增完成")
                st.rerun()

    with c2:
        st.subheader("刪除個案")
        del_labels, del_label_to_id, _ = build_case_label_maps(df)
        del_label = st.selectbox("選擇個案", options=[""] + del_labels)
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
    clicked = out.get("last_clicked")

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

    st.session_state.setdefault("selected_case_ids", [])
    selected = set(st.session_state["selected_case_ids"])

    m = folium.Map(location=[24.07, 120.54], zoom_start=11)
    for _, r in df_ok.iterrows():
        label = f"{r['case_id']}｜{r['name']}"
        color = "green" if r["case_id"] in selected else "blue"
        folium.CircleMarker(
            location=[r["lat"], r["lng"]],
            radius=6,
            tooltip=label,
            popup=f"{r['name']}<br><br>{(r['address_fixed'] or r['address_raw'] or '')}",
            color=color,
            fill=True,
            fill_opacity=0.8
        ).add_to(m)

    c1, c2 = st.columns([2, 1])
    with c1:
        st_folium(m, use_container_width=True, height=650)

    with c2:
        st.subheader("選取個案")
        st.caption(f"起點/終點：{ORIGIN_ADDRESS}")
        st.caption(f"補助：每公里 {SUBSIDY_PER_KM} 元（Google 道路里程）")

        labels, label_to_id, id_to_label = build_case_label_maps(df_ok)
        default_labels = [id_to_label[cid] for cid in selected if cid in id_to_label]
        picked_labels = st.multiselect("已定位個案列表", options=labels, default=default_labels)
        picked = [label_to_id[x] for x in picked_labels]
        picked = sorted(picked, key=case_sort_key)
        st.session_state["selected_case_ids"] = picked

        picked_df = df_ok[df_ok["case_id"].isin(picked)].copy()
        if not picked_df.empty:
            st.dataframe(picked_df[["case_id", "name", "geo_status"]], use_container_width=True)

        st.divider()

        if st.button("🚗 計算路線（Google 最佳化）"):
            if len(picked) < 1:
                st.error("請至少選 1 個個案")
                st.stop()

            if len(picked) > MAX_WAYPOINTS_FOR_DIRECTIONS:
                st.warning(f"你選了 {len(picked)} 個點，先以前 {MAX_WAYPOINTS_FOR_DIRECTIONS} 個計算。")
                picked = picked[:MAX_WAYPOINTS_FOR_DIRECTIONS]
                picked_df = df_ok[df_ok["case_id"].isin(picked)].copy()

            gmaps = googlemaps.Client(key=GOOGLE_KEY)
            points = list(zip(picked_df["lat"].astype(float), picked_df["lng"].astype(float)))
            order, total_m, url = calc_route(gmaps, ORIGIN_ADDRESS, points)

            total_km = total_m / 1000.0
            subsidy = total_km * SUBSIDY_PER_KM

            st.success("計算完成")
            st.metric("道路總里程（km）", f"{total_km:.2f}")
            st.metric("里程補助（元）", f"{subsidy:.0f}")

            if order and len(order) == len(picked_df):
                ordered_df = picked_df.iloc[order].copy()
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
        page = st.radio("功能選單", ["地圖與路線", "Excel 匯入", "個案管理"], index=0)
        if st.button("登出"):
            st.session_state.pop("user", None)
            st.session_state.pop("selected_case_ids", None)
            st.rerun()

        st.divider()
        st.caption("Google API Key 請用環境變數 GOOGLE_MAPS_API_KEY。")

    if page == "Excel 匯入":
        page_import_excel(user)
    elif page == "個案管理":
        page_manage_cases(user)
    else:
        page_map_and_route(user)

main()