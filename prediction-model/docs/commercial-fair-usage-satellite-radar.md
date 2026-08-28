# Commercial Fair Usage, Licensing & Compliance: Himawari-9 Satellite & RainViewer Doppler Radar

*Author: KloudTrack Engineering & Legal Compliance Team*  
*Classification: Public Documentation & Regulatory Compliance*  
*Version: 1.0-MultiModal-2026*

---

## 📑 Executive Summary

The KloudTrack Liquid Neural Network (LNN) prediction engine incorporates multi-modal atmospheric data streams from two primary remote-sensing infrastructure providers:
1. **Himawari-9 Geostationary Meteorological Satellite** (Operated by the Japan Meteorological Agency - JMA).
2. **RainViewer Doppler Weather Radar Network** (Global radar mosaic and nowcast tile service).

This document outlines the **licensing frameworks, commercial usage rights, mandatory attributions, rate limiting policies, and caching rules** governing both data sources to ensure full legal, operational, and commercial compliance.

---

## 1. Himawari-9 Satellite Data (JMA / NOAA Open Data)

### 1.1 Source & Infrastructure Overview
- **Operator**: Japan Meteorological Agency (JMA), Government of Japan.
- **Instrument**: Advanced Himawari Imager (AHI).
- **Distribution Channels**: NOAA Open Data Dissemination (NODD) on Amazon Web Services (AWS S3 `noaa-himawari9`) and JAXA Himawari Monitor.
- **Geographic Coverage**: Asia-Pacific and East Asia, including the entire Philippine Area of Responsibility (PAR).
- **Update Frequency**: Full-disk imagery every 10 minutes; regional rapid scans every 2.5 minutes.

### 1.2 Licensing & Commercial Use Terms
- **Framework**: Governed by the **JMA Data Policy** and the **World Meteorological Organization (WMO) Unified Data Policy (Resolution 1)**.
- **Commercial Rights**: Unrestricted commercial utilization is permitted. KloudTrack is authorized to ingest, process, transform, and incorporate derived parameters (e.g., Cloud Top Brightness Temperature, Convective Cloud Index) into commercial software, municipal dashboards, and paid LGU monitoring subscriptions without royalty obligations.
- **No Warranty / Disclaimer**: Data is provided "as is". JMA and NOAA assume no liability for damages resulting from service interruptions or downstream decisions made using satellite products.

### 1.3 Mandatory Attribution Notice
In compliance with JMA and NOAA open-data guidelines, the platform maintains the following credit in documentation and metadata:
> *"Satellite imagery and cloud dynamics derived from Himawari-9 AHI open data, courtesy of the Japan Meteorological Agency (JMA) and the NOAA Open Data Dissemination (NODD) program."*

### 1.4 Technical Fair Usage & Caching Policy
- **Scan Interval Synchronization**: Ingestion jobs poll satellite metadata at **10-minute intervals**, matching the satellite's native scan cycle.
- **Local Edge Caching**: Processed regional bounding boxes ($14.2^\circ\text{N} - 15.8^\circ\text{N}, 120.0^\circ\text{E} - 121.5^\circ\text{E}$) are cached in memory for a minimum TTL of **600 seconds (10 minutes)** to eliminate redundant network load on public cloud distribution nodes.

---

## 2. RainViewer Doppler Radar API

### 2.1 Service Overview
- **Provider**: RainViewer (MeteoTech LLC).
- **Product**: Global Doppler radar tile mosaics, precipitation reflectivity ($0 - 60\text{ dBZ}$), and 12-frame radar nowcasting.
- **Primary Data Sources**: Aggregates national radar installations, including PAGASA Doppler radar stations (Subic, Tagaytay, Aparri, Baler).

### 2.2 Commercial Licensing Framework
- **Free / Community Tier**: Permitted for non-commercial research, academic benchmarking, and community open-source validation.
- **Commercial Subscription Tier**: Required for commercial deployment, enterprise SLA guarantees, white-labeled municipal dashboards, and high-volume API requests exceeding free-tier rate limits.
- **KloudTrack Compliance**: KloudTrack utilizes authenticated Commercial API access for production endpoints and adheres to the RainViewer Terms of Service.

### 2.3 Mandatory Attribution Notice
All public displays and technical documentation utilizing RainViewer radar data feature standard attribution:
> *"Weather radar and Doppler reflectivity data provided by RainViewer (https://www.rainviewer.com)."*

### 2.4 Technical Fair Usage, Rate Limits & Caching
- **Cache-Control Adherence**: Radar tile assets and metadata maps are cached with an enforced minimum TTL of **600 seconds (10 minutes)**, matching RainViewer's `Cache-Control: public, max-age=600` response header.
- **Exponential Backoff**: Ingestion clients implement exponential backoff ($1\text{s}, 2\text{s}, 4\text{s}$) with circuit breakers to prevent request flooding during network degradation.
- **Tile Coordinate Boundary**: Ingestion is strictly bounded to the Central Luzon tile matrix ($z/x/y$), preventing unnecessary global tile requests.

---

## 3. Compliance Summary Matrix

| Dimension | Himawari-9 Satellite (JMA / NOAA) | RainViewer Doppler Radar API |
| :--- | :--- | :--- |
| **Data Type** | Multispectral Infrared & Cloud Albedo | Dual-Pol Radar Reflectivity ($dBZ$) |
| **Update Interval** | 10 Minutes | 10 Minutes |
| **Commercial Rights** | ✅ Unrestricted Public Open Data | ✅ Permitted via Commercial API License |
| **Required Attribution** | *"JMA via Himawari-9 Open Data / NOAA"* | *"Radar data provided by RainViewer"* |
| **Enforced Cache TTL** | 10 Minutes ($600\text{s}$) | 10 Minutes ($600\text{s}$) |
| **Rate Limit Protocol** | Exponential Backoff on S3/REST | Tiered Token Rate Limiting |
| **Public Safety Disclaimer** | Advisory only; does not supersede PAGASA | Advisory only; does not supersede PAGASA |

---

## 4. Public Safety & Disclaimer Notice

> [!IMPORTANT]
> **Advisory Integration Standard**: Multi-modal satellite and radar inputs are utilized strictly as **supplemental inputs to the KloudTrack LNN prediction engine** to enhance hyper-local nowcasting precision. They do not constitute official statutory meteorological declarations and **do not replace mandatory evacuation advisories issued by PAGASA, NDRRMC, or local disaster authorities**.
