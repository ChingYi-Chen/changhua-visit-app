# 彰化縣社福訪視排程與路線規劃系統 
### Changhua Social Welfare Visit Scheduling System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Deployment](https://img.shields.io/badge/Render-Deployed-brightgreen.svg)](https://render.com/)

## 📌 專案背景與動機
本專案源自於彰化縣政府勞工處個案管理員的實務需求。在處理更生人暨藥癮者訪視工作時，為解決傳統流程中「行政紀錄與地圖導航脫節」的痛點，本系統透過 Python 整合 Google Maps API，實現**一鍵式路徑優化**與**自動化里程計算**，有效提升第一線社福人員的行政效率。

---

## 🚀 線上展示 (Live Demo)
本系統已成功部署於 **Render** 雲端平台，為確保評測流暢，已預載測試資料集供直接使用：
* **系統網頁**：[https://changhua-visit-app.onrender.com]
* **測試帳號**：demo1234
* **測試密碼**：demo1234

---

## 📂 核心檔案功能說明

本專案採模組化設計，實現介面、安全認證與資料邏輯的分離：

### 1. 系統入口與前端
* **`app.py`**：系統主程式，負責 Streamlit UI 渲染、Session 狀態管理及多頁面控制功能。

### 2. 安全認證模組 (Security)
* **`auth.py`**：核心認證模組。整合 `Bcrypt` 雜湊演算法處理密碼，系統不儲存明文密碼，確保使用者資料庫安全，符合資安防護規範。
* **`create_user.py`**：後端管理工具。基於資安原則，系統不開放前台註冊，所有帳號權限皆由管理者透過此腳本於後端安全配發。

### 3. 資料庫與 API 邏輯
* **`db.py`**：資料庫通訊核心（CRUD）。負責個案資料存取及 **地理資訊快取 (Geocoding Cache)** 邏輯，能自動儲存已查詢過的地址與路徑，降低 API 呼叫成本。
* **`requirements.txt`**：定義系統運行所需之環境依賴套件（如 `googlemaps`, `folium`, `bcrypt` 等）。

---

## ✨ 核心技術特色
* **智能路徑優化**：串接 Google Directions API 並啟用 `optimize:true` 參數，自動演算最順路之訪視順序。
* **里程補助結算**：自動根據 Google Maps 回傳之實測里程，依實務標準（每公里 3 元）即時試算補助金額。
* **隱私設計 (Privacy by Design)**：參考 **CIPP/E** 規範，實作帳號間的資料隔離（Data Segregation），確保高敏感個案資訊安全。

---

## ⚙️ 環境配置 (Environment Variables)
本專案之敏感資訊（如 API Key）均透過系統環境變數管理，確保原始碼安全。若欲於本地端執行，需配置：
- `Maps_API_KEY`: Google Cloud Platform 核發之金鑰。

---

### 💡 評測小提醒
1. **建議優先使用網頁版**：建議教授優先點擊上方 **Render 連結** 進行評測，該環境已配置完整的 Google Maps API 運作環境。
2. **資料去識別化**：系統內之個案姓名、地址等資訊均使用模擬數據，僅供功能展示使用。

---
