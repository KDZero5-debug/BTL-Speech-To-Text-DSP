import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft

def analyze_and_stt(wav_path):
    print(f"=== BẮT ĐẦU QUÁ TRÌNH XỬ LÝ TÍN HIỆU: {wav_path} ===")
    
    # ----------------------------------------------------
    # BƯỚC 1: LÝ THUYẾT LẤY MẪU (SAMPLING) & ĐỌC TÍN HIỆU SỐ
    # ----------------------------------------------------
    # Đọc file âm thanh đã được số hóa từ ADC
    fs, signal = wav.read(wav_path)
    
    print(f"[Lấy mẫu] Tần số lấy mẫu thực tế (Fs): {fs} Hz")
    print(f"[Lấy mẫu] Tổng số mẫu thu được (N): {len(signal)} mẫu")
    
    # Kiểm định Định lý Nyquist
    # Giọng nói con người có tần số hữu ích tối đa khoảng Fmax = 4000 Hz
    f_max_speech = 4000 
    print(f"[Kiểm định Nyquist] Fs ({fs} Hz) >= 2 * Fmax ({2 * f_max_speech} Hz): {fs >= 2 * f_max_speech}")
    
    # Nếu âm thanh là Stereo (2 kênh), chuyển về Mono (1 kênh) để xử lý tín hiệu
    if len(signal.shape) > 1:
        signal = signal[:, 0]
        
    # ----------------------------------------------------
    # BƯỚC 2: QUAN SÁT BIÊN ĐỘ & LƯỢNG TỬ HÓA (QUANTIZATION)
    # ----------------------------------------------------
    duration = len(signal) / fs
    print(f"[Thông số] Thời gian đoạn nói: {duration:.2f} giây")
    print(f"[Lượng tử hóa] Biên độ lớn nhất: {np.max(np.abs(signal))}")

    # ----------------------------------------------------
    # BƯỚC 3: PHÂN TÍCH TẦN SỐ (BIẾN ĐỔI FOURIER - DFT/FFT)
    # Đây là bước cốt lõi để trích xuất đặc trưng âm thanh nhằm nhận diện từ
    # ----------------------------------------------------
    # Lấy một đoạn tín hiệu ngắn (Khung tín hiệu - Windowing) để phân tích tần số
    # Vì đặc tính tiếng nói thay đổi theo thời gian (Non-stationary)
    window_size = int(0.03 * fs) # Khung 30ms theo lý thuyết tiếng nói
    frame = signal[:window_size]
    
    # Áp dụng hàm cửa sổ Hamming để giảm hiện tượng rò rỉ phổ (Spectral Leakage)
    hamming_window = np.hamming(window_size)
    windowed_frame = frame * hamming_window
    
    # Thực hiện Biến đổi Fourier nhanh (FFT) để chuyển từ miền Thời gian -> Miền Tần số
    fft_spectrum = np.abs(fft(windowed_frame))
    frequencies = np.fft.fftfreq(window_size, 1/fs)
    
    # Chỉ lấy nửa dương của phổ tần số (đối xứng)
    positive_freq_idx = np.where(frequencies >= 0)
    main_freqs = frequencies[positive_freq_idx]
    main_spectrum = fft_spectrum[positive_freq_idx]
    
    # Tìm tần số đỉnh (Tần số trội nhất trong đoạn âm thanh)
    peak_freq = main_freqs[np.argmax(main_spectrum)]
    print(f"[Trích xuất đặc trưng] Tần số trội nhất phát hiện được: {peak_freq:.2f} Hz")

    # ----------------------------------------------------
    # BƯỚC 4: RA QUYẾT ĐỊNH (NHẬN DIỆN VĂN BẢN ĐƠN GIẢN)
    # Mô phỏng bộ phân lớp ký tự dựa trên tần số cơ bản đặc trưng
    # ----------------------------------------------------
    print("\n--- KẾT QUẢ CHUYỂN LỜI NÓI THÀNH VĂN BẢN (STT) ---")
    
    # Ví dụ giả lập nhận diện dựa trên các dải tần số của các nguyên âm
    # (Trong thực tế AI sẽ dùng ma trận MFCC và học máy, ở đây ta dùng logic DSP cơ bản)
    if 80 <= peak_freq <= 150:
        result = "[VĂN BẢN]: GIỌNG NAM (Tần số cơ bản thấp)"
    elif 150 < peak_freq <= 250:
        result = "[VĂN BẢN]: GIỌNG NỮ (Tần số cơ bản cao)"
    elif 250 < peak_freq <= 1000:
        result = "[VĂN BẢN]: TỪ 'A' HOẶC 'O' (Tần số Formant thấp)"
    else:
        result = "[VĂN BẢN]: KHÔNG XÁC ĐỊNH ĐƯỢC TỪ (Nhiễu tần số cao hoặc âm gió)"
        
    print(result)
    print("==================================================\n")
    return result

if __name__ == "__main__":
    # Test thử nghiệm hệ thống bằng cách tạo một file tín hiệu mẫu giả lập hình sin
    # Giả lập một người đang nói tạo ra sóng âm có tần số 200 Hz (Giọng nữ), được lấy mẫu ở Fs = 16000 Hz
    fs_test = 16000
    t = np.linspace(0, 1, fs_test, endpoint=False)
    simulated_voice = np.sin(2 * np.pi * 200 * t) * 3000  # Biên độ lượng tử hóa
    
    # Ghi thành file .wav tạm thời để chạy kiểm thử sơ đồ khối
    test_filename = "test_giong_noi.wav"
    wav.write(test_filename, fs_test, simulated_voice.astype(np.int16))
    
    # Chạy hàm STT bằng lý thuyết DSP vừa lập trình
    analyze_and_stt(test_filename)
