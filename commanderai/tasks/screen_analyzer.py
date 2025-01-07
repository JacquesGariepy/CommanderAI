#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyse de l'écran : capture (mss), OCR (pytesseract), détection UI.
"""
import os
import logging
import asyncio
import mss
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional

logging.basicConfig(level=logging.DEBUG)

TESSERACT_CONFIG = r'--oem 3 --psm 6'
MIN_ELEMENT_WIDTH = 20
MAX_ELEMENT_WIDTH = 300
MIN_ELEMENT_HEIGHT = 20
MAX_ELEMENT_HEIGHT = 100

class ScreenAnalyzer:
    def __init__(self):
        logging.info("[ScreenAnalyzer] init")
        self.screen_capture = mss.mss()
        self.screen_capture_lock = asyncio.Lock()

    async def capture_screen(self) -> Optional[np.ndarray]:
        try:
            async with self.screen_capture_lock:
                raw = await asyncio.to_thread(self._do_capture)
                if raw and hasattr(raw, 'rgb') and raw.rgb:
                    img = Image.frombytes('RGB', raw.size, raw.rgb)
                    return np.array(img)
        except Exception as e:
            logging.error(f"[ScreenAnalyzer] capture_screen => {e}")
        return None

    def _do_capture(self):
        try:
            with mss.mss() as sct:
                return sct.grab(sct.monitors[0])
        except Exception as e:
            logging.error(f"[ScreenAnalyzer] _do_capture => {e}")
            return None

    def analyze_screen(self) -> Tuple[str, List[Dict[str, Any]]]:
        try:
            screen_img = asyncio.run(self.capture_screen())
            if screen_img is None:
                return "", []
            text = self._extract_text(screen_img)
            elements = self._detect_ui_elements(screen_img)
            return text, elements
        except Exception as e:
            logging.error(f"[ScreenAnalyzer] analyze_screen => {e}")
            return "", []

    def _extract_text(self, image: np.ndarray) -> str:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            denoised = cv2.medianBlur(thresh, 3)
            kernel = np.ones((2,2), np.uint8)
            dilated = cv2.dilate(denoised, kernel, iterations=1)
            txt = pytesseract.image_to_string(dilated, config=TESSERACT_CONFIG, lang='fra')
            return txt.strip()
        except Exception as e:
            logging.error(f"[ScreenAnalyzer] _extract_text => {e}")
            return ""

    def _detect_ui_elements(self, image: np.ndarray) -> List[Dict[str, Any]]:
        output = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < 100:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if MIN_ELEMENT_WIDTH < w < MAX_ELEMENT_WIDTH and MIN_ELEMENT_HEIGHT < h < MAX_ELEMENT_HEIGHT:
                    roi = gray[y:y+h, x:x+w]
                    el_type = self._classify(roi, w, h)
                    conf = self._confidence(roi)
                    if conf > 30:
                        output.append({
                            "type": el_type,
                            "position": (x, y),
                            "size": (w, h),
                            "center": (x + w//2, y + h//2),
                            "confidence": conf
                        })
        except Exception as e:
            logging.error(f"[ScreenAnalyzer] _detect_ui_elements => {e}")
        return output

    def _classify(self, roi: np.ndarray, w: int, h: int) -> str:
        try:
            aspect = float(w)/float(h) if h!=0 else 0
            stddev = np.std(roi)
            if 2.5 < aspect < 8 and stddev > 30:
                return "text_field"
            elif 0.8 < aspect < 1.2:
                return "button"
            elif aspect > 8:
                return "menu"
            else:
                return "unknown"
        except Exception:
            return "unknown"

    def _confidence(self, roi: np.ndarray) -> float:
        if roi.size == 0:
            return 0.0
        std_dev = np.std(roi)
        mean_val = np.mean(roi)
        base = (std_dev / 128.0)*100
        mean_factor = (mean_val/255.0)*0.5 + 0.5
        return float(min(100.0, max(0.0, base*mean_factor)))
