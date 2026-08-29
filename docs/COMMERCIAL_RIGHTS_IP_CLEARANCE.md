# Commercial Rights, IP Clearance & Fair Usage Documentation

**Organization:** Kloudtech Inc. / KloudTrack Hydrometeorological Intelligence  
**Document Type:** Intellectual Property Clearance, Academic Attribution & Statutory Fair Use Declaration  
**Scope:** Citizendashboard, PINN-LNN Continuous-Time Prediction Engine & Real-Time Telemetry Pipeline  
**Version:** 2.4-Production  
**Effective Date:** August 2026  

---

## 1. Executive Summary & Commercial Freedom-to-Operate (FTO)

Kloudtech Inc. maintains **full, unencumbered commercial rights** to deploy, monetize, license, and distribute the **Citizendashboard** platform and the underlying **Physics-Informed Liquid Neural Network (PINN-LNN)** engine. 

1. **Zero Proprietary Third-Party Algorithm Restrictions**: The mathematical architecture of Liquid Neural Networks and Physics-Informed Neural ODEs operates on peer-reviewed, open scientific foundations published in the public academic domain.
2. **Proprietary Neural Weights & Codebase**: All neural network weight tensors, ODE state transition matrices, TypeScript service implementations, spatial IDW algorithms, and continuous-time prediction rollouts are 100% original, proprietary creations owned by Kloudtech Inc.
3. **Hardware Telemetry Ownership**: Real-time telemetry is sourced directly from Kloudtrack-owned Automated Weather Stations (AWS) and Water Level Monitoring Stations (WLMS) via private AWS IoT Core MQTT mTLS brokers.
4. **No Third-Party Commercial Data Dependencies**: The production application operates independently without commercial lock-in, recurring third-party API data fees, or restrictive vendor terms.

---

## 2. Scientific Academic Attributions

In accordance with standard academic and scientific citation conventions, the following foundational scientific literature is credited for introducing the theoretical principles utilized in this engine:

### 2.1 Liquid Time-Constant (LTC) & Closed-Form Continuous-Time (CfC) Networks
- **Theoretical Origin**: Developed by Dr. Ramin Hasani, Dr. Mathias Lechner, Dr. Alexander Amini, Prof. Daniela Rus, and collaborators at the **MIT Computer Science and Artificial Intelligence Laboratory (MIT CSAIL)** and TU Wien.
- **Key Publications**:
  - Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). *"Liquid Time-constant Networks."* **Nature Machine Intelligence**, 3(2), 148–160. DOI: [10.1038/s42256-020-00287-3](https://doi.org/10.1038/s42256-020-00287-3).
  - Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G., & Rus, D. (2022). *"Closed-form continuous-time neural networks."* **Nature Machine Intelligence**, 4(11), 992–1003. DOI: [10.1038/s42256-022-00556-7](https://doi.org/10.1038/s42256-022-00556-7).
- **Attribution & Usage Scope**: The mathematical closed-form differential equations are implemented independently in native TypeScript and Python under open scientific principles. No proprietary closed-source binaries from third parties are included.

### 2.2 Neural Ordinary Differential Equations (Neural ODEs)
- **Theoretical Origin**: Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *"Neural ordinary differential equations."* **Advances in Neural Information Processing Systems (NeurIPS 2018)**, 31.

### 2.3 Physics-Informed Neural Networks (PINNs)
- **Theoretical Origin**: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations."* **Journal of Computational Physics**, 378, 686–707.

---

## 3. Open Source Software (OSS) Licenses

All third-party open source libraries utilized in this project are distributed under permissive licenses allowing commercial use, modification, distribution, and private deployment:

| Software / Package | License | Permitted Commercial Usage | Copyright / Source |
| :--- | :---: | :---: | :--- |
| **Next.js** | MIT | ✅ Commercial Use Permitted | © Vercel, Inc. |
| **React / React-DOM** | MIT | ✅ Commercial Use Permitted | © Meta Platforms, Inc. |
| **Tailwind CSS** | MIT | ✅ Commercial Use Permitted | © Tailwind Labs, Inc. |
| **Lucide React** | ISC / MIT | ✅ Commercial Use Permitted | © Lucide Contributors |
| **Recharts** | MIT | ✅ Commercial Use Permitted | © Recharts Group |
| **next-intl** | MIT | ✅ Commercial Use Permitted | © Jan Amann |
| **PyTorch** | Modified BSD | ✅ Commercial Use Permitted | © The Linux Foundation / PyTorch Contributors |
| **NumPy** | BSD-3-Clause | ✅ Commercial Use Permitted | © NumPy Developers |

---

## 4. Fair Use & Non-Consumptive Training Clause

### 4.1 Purpose of Public Weather & Hydrological Data
During model research, offline optimization, and verification benchmarking, publicly accessible hydrometeorological observations (e.g. World Meteorological Organization [WMO] synoptic station records, PAGASA flood bulletins, and JMA Himawari-9 satellite infrared indices) are referenced strictly for:
1. **Non-Consumptive Scientific Training**: Evaluating loss function gradients during offline evolutionary weight tournaments.
2. **Ground-Truth Statistical Validation**: Benchmarking Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and F1 detection scores against standardized meteorological benchmarks.

### 4.2 Statutory Fair Use Declaration
This usage strictly conforms to statutory Fair Use principles (17 U.S.C. § 107 and Section 185 of Republic Act No. 8293 / Philippine IP Code) under the following four legal pillars:

1. **Transformative Purpose**: Raw observations are transformed into high-order continuous-time ODE dynamic coefficients ($\mathbf{W}_{\text{in}}, \mathbf{W}_{\text{rec}}, \boldsymbol{\tau}$). The original data is not reproduced, stored, redistributed, or repackaged.
2. **Nature of the Work**: Factual, public-domain atmospheric measurements and planetary thermodynamic constants.
3. **Amount and Substantiality**: Only statistical aggregated scalar quantities were referenced as boundary loss constraints.
4. **Market Effect**: The platform does not compete with or displace public agency meteorological services; rather, it delivers hyper-localized, private-network edge nowcasting for community flood resilience and commercial IoT asset monitoring.

---

## 5. Summary of Commercial Compliance

| Compliance Dimension | Status | Verification Detail |
| :--- | :---: | :--- |
| **Commercial Rights** | **100% CLEAR** | Full commercial freedom-to-operate without royalties |
| **Patent / Algorithm Freedom** | **100% CLEAR** | Open mathematical equations implemented in native source code |
| **Data Provenance** | **100% CLEAR** | Live telemetry produced by Kloudtrack-owned physical IoT stations |
| **OSS License Compliance** | **100% CLEAR** | All client/server packages operate under MIT / BSD / ISC licenses |
| **Academic Attribution** | **100% COMPLETE** | Hasani et al., Raissi et al., and Chen et al. properly attributed |
