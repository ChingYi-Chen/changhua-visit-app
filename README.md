# 彰化縣社福訪視排程與路線規劃系統 
### Changhua Social Welfare Visit Scheduling System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![GDPR Compliant](https://img.shields.io/badge/Privacy-GDPR_Compliant-green.svg)](#-data-privacy--security-gdpr-alignment)

## 📌 Project Overview
Developed by a professional Case Manager at the **Changhua County Government**, this system is a decision support tool designed to bridge the gap between administrative Excel records and field navigation for social workers. 

By integrating **Python** with **Google Maps APIs**, it automates route optimization and travel subsidy calculations, transforming a labor-intensive manual process into a streamlined digital workflow.

---

## 🚀 Live Demo
* **Web Link**: [https://changhua-visit-app.onrender.com]
* **Test Account**: `demo1234`
* **Test Password**: `demo1234`
> *Note: It is recommended to use the web version for evaluation as the API environment is pre-configured.*

---

## 🛡️ Data Privacy & Security: GDPR Alignment
The system is built on **Privacy by Design** principles, aligning technical implementations with the European **General Data Protection Regulation (GDPR)** standards to protect sensitive case data.

### 1. Data Minimization & Purpose Limitation
* **GDPR Principle**: Only process data "necessary" for a specific purpose.
* **Implementation**: The system uses a **Pass-through Filtering** mechanism. Although social work Excel files contain highly sensitive data (e.g., medical or criminal history), the system is hard-coded to **extract only three necessary fields: Case ID, Name, and Address**. All other sensitive columns are ignored at the memory level and never stored.

### 2. Privacy by Default
* **GDPR Principle**: Privacy protections are embedded into the lifecycle, not added as an afterthought.
* **Implementation**: A closed-loop architecture is used. Public registration is disabled to prevent unauthorized access. It utilizes **Bcrypt hashing** for passwords and a **Geocoding Cache** to minimize the "data footprint" exchanged with third-party APIs.

### 3. Data Segregation & Integrity
* **GDPR Principle**: Ensure protection against unauthorized access.
* **Implementation**: Implements **Physical Data Segregation** via SQLite. Each manager only accesses their own imported cases. API keys are managed through server environment variables, never hard-coded in the repository.

---

## 🛠️ Technical Architecture & Database Design

### 1. System Logic
The application utilizes **Streamlit** for the frontend, while the backend handles geospatial logic and API communication.
* **Data Layer**: SQLite manages user isolation and geocoding caches.
* **Computation Layer**: Implements the **Held–Karp (Dynamic Programming) algorithm** to solve the Traveling Salesman Problem (TSP).



### 2. Database Schema
* **`users`**: Stores Bcrypt-hashed credentials.
* **`cases`**: Manages case coordinates and geocoding status (`OK`, `FAIL`, `MANUAL`), allowing for reverse-geocoding via map pin-drops.
* **`geocode_cache` & `distance_cache`**: Reduces API costs by storing previously resolved addresses and road-distance matrices.

### 3. Route Optimization Logic
* **Global Optimization**: Unlike simple greedy algorithms, this system solves for the **absolute shortest loop** using DP.
* **Real-world Distance**: Calculations are based on actual road distance (Google Distance Matrix) rather than straight-line distance, ensuring that the **3 TWD/KM** subsidy calculation matches real-world fuel consumption.



---

## ✨ Key Features
* **Auto-Header Detection**: The `find_header_row` function intelligently identifies the correct data starting point in complex Excel files.
* **Interactive Map Correction**: Integrated **Folium** maps allow users to correct inaccurate addresses by manually dropping a pin on the map.
* **Natural Sort**: Case IDs are sorted naturally (e.g., A2 < A10) to match administrative logic.
* **One-Click Navigation**: Generates direct Google Maps navigation links for field use.

---

## 📂 File Structure
* `app.py`: Main entry point (UI rendering, Session management, Multi-page logic).
* `auth.py`: Security module (Bcrypt hashing and authentication).
* `db.py`: Database communication core (CRUD and Schema definitions).
* `create_user.py`: Administrative tool for secure account provisioning.

---

### 💡 Developer's Statement
As both a social worker and a developer, I recognize that privacy for vulnerable populations is paramount. This system is not just about route optimization; it is a technical defense line designed to protect human dignity through compliant digital transformation.
