# 彰化縣社福訪視排程與路線規劃系統 
### Changhua Social Welfare Visit Scheduling System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 專案背景與動機
本專案源自於彰化縣政府勞工處個案管理員的實務需求。在處理更生人暨藥癮者訪視工作時，傳統流程仰賴 Excel 紀錄與人工手動輸入 Google Maps，存在以下痛點：
- **行程規劃耗時**：需反覆手動調整路徑。
- **地址定位困難**：不完整地址難以精確定位。
- **補貼計算繁瑣**：里程與交通補助需人工估算。

本系統透過 Python 整合 Google Maps API，實現**一鍵式路徑優化**與**自動化里程計算**，將行政效率提升至最大化。

---

## 🚀 線上展示 (Live Demo)
本系統已部署於 Render 雲端平台，歡迎點擊下方連結體驗：
* **系統網頁**：[請填入你的 Render 網址]
* **測試帳號**：`填入你設定的帳號`
* **測試密碼**：`填入你設定的密碼`

---

## 🛠️ 核心功能
1. **智能路徑優化**：串接 Google Directions API，利用 `optimize:true` 演算法自動排列最佳訪視順序。
2. **個案資料管理**：支援 Excel 批次匯入，並建立每位使用者獨立的 SQLite 資料倉儲。
3. **地理資訊快取 (Caching)**：設計 Geocoding 與 Distance Cache，減少重複呼叫 API，提升響應速度並降低運作成本。
4. **里程補助結算**：自動計算路徑總里程，並按實務標準（每公里 3 元）估算補助費用。

---

## 🔒 資安與隱私設計 (Privacy by Design)
本專案嚴格遵循隱私設計原則，確保高敏感個資安全：
- **帳號隔離**：不同帳號間資料庫完全獨立，互不隸屬。
- **Bcrypt 加密**：使用者密碼經雜湊處理，不以明文儲存。
- **金鑰防護**：Google Maps API Key 透過系統環境變數管理，不留存於代碼中。
- **去識別化**：示範資料集已進行去識別化處理。

---

## ⚙️ 技術架構
- **前端**：Streamlit
- **資料庫**：SQLite
- **地圖整合**：Folium / Streamlit-folium
- **地理運算**：Google Maps Platform (Geocoding, Directions, Distance Matrix)
