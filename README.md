# OpenSim Converter（單人版）

Windows 圖形介面工具：選擇 MP4 影片或 CSV、OpenSim `.osim` 模型和輸出資料夾，產生 CSV 與模型 marker TRC。

## 老師下載與使用

請到這個 GitHub 專案的 **Releases**，下載附件 **OpenSimConverter_老師版.zip**。

1. 對 ZIP 按右鍵，選「全部解壓縮」。
2. 開啟 `OpenSimConverter` 資料夾，執行 `OpenSimConverter.exe`。
3. 選擇影片或 CSV、模型和輸出位置，按「開始轉換」。

執行檔已包含 Python、MediaPipe、OpenCV 與 OpenSim Python 套件，不需要自行安裝。搬移時請保留完整資料夾與 `_internal`。

GitHub 自動提供的 **Source code (zip)** 是原始碼，沒有可直接使用的 EXE。老師應下載上面指定的老師版附件。

完整操作與錯誤排查：[使用說明](使用說明.txt)。

## 原始碼

| 檔案 | 用途 |
| --- | --- |
| `OpenSimConverter.py` | 選檔介面、單人轉換與 CSV marker 擴增整合 |
| `VideoToCSV.py` | 影片偵測、CSV 清理、marker 擴增及 TRC 輸出 |
| `CSVToTRC.py` | CSV 座標自動貼合模型與 TRC 輸出 |
| `native_paths.py` | Windows 中文模型路徑相容處理 |
| `pose_landmarker_full.task` | 隨程式提供的 MediaPipe Pose 偵測模型 |
| `requirements-lock.txt` | 已驗證的相依套件版本 |
| `build.ps1` | Windows EXE 打包指令 |

本工具只處理一人。CSV 人物編號與 FPS 會自動讀取；輸入 CSV 必須只含一人。全身模型需含相容的 MarkerSet，目前驗證模型為原使用的 FullBodyModel（83 個 markers）。工具輸出 TRC；IK 與肌肉分析需另外在 OpenSim 執行。

## 從原始碼執行或打包

需要 Windows 64 位元和 Python 3.12。在原始碼資料夾開啟 PowerShell：

```powershell
py -3.12 -m venv .build-venv
& '.build-venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
& '.build-venv\Scripts\python.exe' OpenSimConverter.py
```

重新打包：

```powershell
.\build.ps1
```

成品會產生在 `dist\OpenSimConverter`。發布時把這個資料夾壓縮成 ZIP，附加到 GitHub Release。

`.gitignore` 已排除 EXE、打包套件、轉換結果和建置環境，讓程式碼倉庫保持精簡。
