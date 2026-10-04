"""Portable desktop frontend for the existing single-person pipeline."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import subprocess
import sys

import traceback
from datetime import datetime


def resource(name):
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / name


def convert_csv(source, model, output, person, rate):
    import numpy as np
    import VideoToCSV as dance
    import CSVToTRC as converter

    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = [r for r in reader if int(r.get("person") or r.get("hand") or 0) == person]
    if not rows:
        raise ValueError(f"CSV 沒有 person/hand={person} 的資料")
    names, positions = dance.load_model_markers(model)
    if "marker" in fields:
        # Already expanded: retain calibrated coordinates without fitting twice.
        actual = set(r["marker"] for r in rows)
        unknown = actual - set(names)
        if unknown:
            raise ValueError("CSV marker 與所選模型不符：" + ", ".join(sorted(unknown)))
        order = [n for n in names if n in actual]
        frames = sorted(set(int(r["frame"]) for r in rows))
        frame_map = {f: i for i, f in enumerate(frames)}
        marker_map = {n: i for i, n in enumerate(order)}
        values = np.full((len(frames), len(order), 3), np.nan)
        times = np.full(len(frames), np.nan)
        for row in rows:
            if row.get("units", "m").lower() != "m":
                raise ValueError("marker CSV 的單位必須是 m")
            i = frame_map[int(row["frame"])]
            times[i] = float(row["time"])
            values[i, marker_map[row["marker"]]] = [float(row[a]) for a in "xyz"]
        if not np.all(np.isfinite(values)):
            raise ValueError("marker CSV 有缺失或無效座標，請改用 clean.csv 重建")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("CSV 時間必須遞增")
        fps = float(rate) if rate else (1 / float(np.median(np.diff(times))) if len(times) > 1 else 30)
    else:
        data = converter.read_csv_data(source)
        if data.kind != "pose":
            args = [str(source), "-o", str(output / f"{source.stem}.trc"),
                    "--fit-model", str(model), "--fit-index", str(person)]
            if rate:
                args += ["--data-rate", str(rate)]
            if converter.main(args):
                raise ValueError("手部 CSV 轉換失敗，請查看紀錄")
            return
        if data.coordinate_space != "world":
            raise ValueError("全身 CSV 必須使用 MediaPipe world 座標")
        keys = [k for point in data.points.values() for k in point if k.hand == person and k.kind == "pose"]
        if not keys:
            raise ValueError(f"沒有 person={person} 的 pose landmarks")
        frames = sorted(data.times)
        times = np.array([data.times[f] for f in frames], dtype=float)
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("CSV 需要至少兩個時間遞增的影格")
        fps = float(rate) if rate else float(converter.infer_data_rate(data.times))
        points = np.full((len(frames), 33, 3), np.nan)
        for i, frame in enumerate(frames):
            for key, point in data.points.get(frame, {}).items():
                if key.hand == person and key.kind == "pose":
                    points[i, key.landmark] = point
        count = len(frames)
        series = dance.PoseSeries(person, np.arange(count), times, points,
            np.ones((count, 33)), np.full((count, 33, 2), np.nan),
            np.full((count, 4), np.nan), np.ones(count), np.zeros(count, dtype=bool),
            np.zeros((count, 33), dtype=int))
        pose, summary = dance.fit_pose_to_model(series, model, 1)
        order, templates = dance.create_marker_templates(names, positions)
        values = dance.augment_markers(pose, order, templates)
        values, spikes = dance.filter_marker_spikes(values, fps,
            max_speed=8.0, max_acceleration=120.0)
        values, _ = dance.apply_ground_correction(values, order, positions)
        values, _ = dance.align_markers_to_model_reference(values, order, positions)
        print(f"模型貼合 RMS: {summary.calibration_rms:.6g} m；移除尖峰：{spikes}", flush=True)
    missing = set(names) - set(order)
    if missing:
        print("提醒：這個模型有未支援的 markers：" + ", ".join(sorted(missing)), flush=True)
    marker_csv = output / f"{source.stem}_model_markers.csv"
    dance.write_marker_csv(marker_csv, person, times, order, values)
    trc = output / f"{source.stem}.trc"
    dance.write_trc(trc, fps, times, order, values)
    print(f"完成：{len(times)} 幀、{len(order)}/{len(names)} 模型 markers、{fps:.6g} Hz\nTRC：{trc}", flush=True)


def run_worker(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    # Windowed EXEs have no stdout. Log to disk for both GUI and CLI workers.
    with (output / "conversion.log").open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = sys.stderr = log
        try:
            print("開始轉換；只處理一個人。", flush=True)
            import VideoToCSV as dance
            source, model = Path(config["input"]), Path(config["model"])
            dance.load_model_markers(model)
            if config["mode"] == "video":
                args = [str(source), "--fit-model", str(model), "--output-dir", str(output),
                        "--model", str(resource("pose_landmarker_full.task")), "--num-people", "1",
                        "--progress-every", "25"]
                if config.get("red"):
                    args += ["--target-shirt-color", "red"]
                if dance.main(args):
                    raise ValueError("影片轉換失敗，請查看上述紀錄")
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                for item in manifest["outputs"]:
                    if not Path(item["trc"]).is_file():
                        raise ValueError("TRC 沒有成功產生")
            else:
                convert_csv(source, model, output, config["person"], config.get("rate"))
            print("SUCCESS：轉換完成", flush=True)
            return 0
        except Exception:
            traceback.print_exc()
            return 1


def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    root = tk.Tk()
    root.title("OpenSim 影片 / CSV 轉換工具（單人版）")
    root.geometry("860x660")
    root.minsize(760, 580)
    style = ttk.Style(root)
    style.configure("TLabel", font=("Microsoft JhengHei", 10))
    style.configure("TButton", font=("Microsoft JhengHei", 10))
    panel = ttk.Frame(root, padding=18)
    panel.pack(fill="both", expand=True)
    panel.columnconfigure(1, weight=1)
    panel.rowconfigure(10, weight=1)
    mode = tk.StringVar(value="video")
    source = tk.StringVar()
    model = tk.StringVar()
    destination = tk.StringVar()
    status = tk.StringVar(value="請選擇影片、模型和輸出位置。")
    state = {"process": None, "output": None, "log_pos": 0}
    ttk.Label(panel, text="影片 → CSV → TRC　／　已有 CSV → TRC", font=("Microsoft JhengHei", 16, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,12))
    modes = ttk.Frame(panel)
    modes.grid(row=1, column=0, columnspan=3, sticky="w", pady=8)
    ttk.Radiobutton(modes, text="選影片（只偵測一人）", variable=mode, value="video").pack(side="left")
    ttk.Radiobutton(modes, text="選已有的 CSV", variable=mode, value="csv").pack(side="left", padx=20)

    def choose_input():
        types = [("MP4 影片", "*.mp4")] if mode.get() == "video" else [("CSV", "*.csv")]
        path = filedialog.askopenfilename(title="選擇輸入檔案", filetypes=types)
        if path:
            source.set(path)

    for row, label, var, choose in [
        (2, "影片 / CSV", source, choose_input),
        (3, "OpenSim 模型", model, lambda: model.set(filedialog.askopenfilename(title="選擇含 MarkerSet 的模型", filetypes=[("OpenSim 模型", "*.osim")]) or model.get())),
        (4, "輸出資料夾", destination, lambda: destination.set(filedialog.askdirectory(title="選擇輸出位置") or destination.get()))]:
        ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=7)
        ttk.Entry(panel, textvariable=var).grid(row=row, column=1, sticky="ew", padx=10)
        ttk.Button(panel, text="瀏覽…", command=choose).grid(row=row, column=2)
    ttk.Label(panel, text="模型需含相容的身體 markers；每次轉換會建立獨立子資料夾。\n影片模式會輸出 raw / clean / model markers CSV 和 TRC。CSV 模式支援上述 CSV。").grid(row=6, column=0, columnspan=3, sticky="w", pady=5)
    buttons = ttk.Frame(panel)
    buttons.grid(row=7, column=0, columnspan=3, sticky="w", pady=10)
    progress = ttk.Progressbar(panel, mode="indeterminate")
    progress.grid(row=8, column=0, columnspan=3, sticky="ew", pady=5)
    ttk.Label(panel, textvariable=status, wraplength=790).grid(row=9, column=0, columnspan=3, sticky="w", pady=5)
    logbox = tk.Text(panel, font=("Microsoft JhengHei",9), wrap="word", height=13, state="disabled")
    logbox.grid(row=10, column=0, columnspan=3, sticky="nsew")

    def append(text):
        logbox.configure(state="normal")
        logbox.insert("end", text)
        logbox.see("end")
        logbox.configure(state="disabled")

    def start():
        try:
            src = Path(source.get()).expanduser().resolve()
            mdl = Path(model.get()).expanduser().resolve()
            if not source.get() or not src.is_file() or src.suffix.lower() != (".mp4" if mode.get() == "video" else ".csv"):
                raise ValueError("請選擇正確的 MP4 / CSV 檔案")
            if not model.get() or not mdl.is_file() or mdl.suffix.lower() != ".osim":
                raise ValueError("請選擇 .osim 模型")
            if not destination.get():
                raise ValueError("請選擇輸出資料夾")
            fps = None
            pid = 0
            if mode.get() == "csv":
                with src.open(encoding="utf-8-sig", newline="") as handle:
                    ids = sorted(set(int(r.get("person") or r.get("hand") or 0) for r in csv.DictReader(handle)))
                if not ids:
                    raise ValueError("CSV 沒有資料")
                if len(ids) != 1:
                    raise ValueError(f"請使用只含一人的 CSV（目前人物編號：{ids}）")
                pid = ids[0]
            output = Path(destination.get()).expanduser().resolve() / (src.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            output.mkdir(parents=True, exist_ok=False)
            config = output / "settings.json"
            config.write_text(json.dumps(dict(mode=mode.get(), input=str(src), model=str(mdl), output=str(output), person=pid, rate=fps, red=False), ensure_ascii=False, indent=2), encoding="utf-8")
            if getattr(sys, "frozen", False):
                command = [sys.executable, "--worker", str(config)]
            else:
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(config)]
            state.update(process=subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW), output=output, log_pos=0)
            start_button.configure(state="disabled")
            cancel_button.configure(state="normal")
            progress.start(12)
            status.set("轉換中，請稍候…")
            append(f"\n輸出：{output}\n")
        except Exception as exc:
            messagebox.showerror("無法開始", str(exc))

    def cancel():
        proc = state["process"]
        if proc and proc.poll() is None:
            proc.terminate()
            status.set("已停止；部分輸出檔可能尚未完成。")

    def open_output():
        if state["output"]:
            os.startfile(str(state["output"]))

    start_button = ttk.Button(buttons, text="開始轉換", command=start)
    start_button.pack(side="left")
    cancel_button = ttk.Button(buttons, text="停止", command=cancel, state="disabled")
    cancel_button.pack(side="left", padx=10)
    ttk.Button(buttons, text="開啟輸出資料夾", command=open_output).pack(side="left")

    def poll():
        proc = state["process"]
        if proc:
            path = state["output"] / "conversion.log"
            if path.exists():
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(state["log_pos"])
                    text = handle.read()
                    state["log_pos"] = handle.tell()
                if text:
                    append(text)
            code = proc.poll()
            if code is not None:
                state["process"] = None
                progress.stop()
                start_button.configure(state="normal")
                cancel_button.configure(state="disabled")
                if code == 0:
                    status.set("完成！請按「開啟輸出資料夾」查看 CSV 和 TRC。")
                    messagebox.showinfo("轉換完成", f"檔案已儲存至：\n{state['output']}")
                else:
                    status.set("未完成，請查看紀錄；conversion.log 已存於輸出資料夾。")
        root.after(500, poll)

    def close():
        if state["process"] and state["process"].poll() is None:
            if not messagebox.askyesno("正在轉換", "關閉會停止轉換，要關閉嗎？"):
                return
            cancel()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    poll()
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(run_worker(sys.argv[2]))
    launch_gui()

