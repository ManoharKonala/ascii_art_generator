#!/usr/bin/env python3
"""
Ultimate ASCII Art Generator
Includes all professional techniques PLUS color, artistic styles, and creative modes
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFont, ImageDraw
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from typing import List, Tuple, Dict, Optional
import argparse
import os

class UltimateASCIIGenerator:
    def __init__(self):
        # All original professional character sets (UNCHANGED)
        self.calibrated_character_set = "$@B%8&#WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
        
        self.character_sets = {
            'professional': "$@B%8&#WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
            'ultra_sharp': "@#S%?*+;:,. ",
            'blocks_pro': "█▓▒░▄▀ ",
            'classic_pro': "@%#*+=-:. ",
            
            # FIXED ARTISTIC CHARACTER SETS - Properly ordered by visual density
            'artistic_blocks': "██▓▒░▄▀▐▌▊▋ ",
            'double_width': "██▉▊▋▌▍▎▏ ",
            'shaded_blocks': "█▓▒░ ",
            'fine_blocks': "█▉▊▋▌▍▎▏▁▂▃▄▅▆▇ ",
            'mixed_symbols': "█▓▒░■□▪▫●○◐◑◒◓ ",
            'custom': ""
        }
        
        # FIXED: Simplified and working color palettes
        self.color_palettes = {
            'grayscale': [(255, 255, 255), (192, 192, 192), (128, 128, 128), (64, 64, 64), (0, 0, 0)],
            'rainbow': [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (75, 0, 130), (238, 130, 238)],
            'fire': [(255, 255, 255), (255, 255, 0), (255, 165, 0), (255, 0, 0), (139, 0, 0)],
            'ocean': [(255, 255, 255), (135, 206, 235), (70, 130, 180), (0, 0, 139), (0, 0, 0)],
            'forest': [(255, 255, 255), (144, 238, 144), (34, 139, 34), (0, 100, 0), (0, 0, 0)],
            'sunset': [(255, 255, 255), (255, 218, 185), (255, 165, 0), (220, 20, 60), (0, 0, 0)]
        }
        
        # Enhanced character density mapping with new block characters
        self.character_densities = {
            ' ': 0.0, '.': 0.05, '`': 0.08, '^': 0.1, '"': 0.12, ',': 0.15,
            ':': 0.18, ';': 0.2, 'I': 0.22, 'l': 0.24, '!': 0.26, 'i': 0.28,
            '>': 0.3, '<': 0.32, '~': 0.34, '+': 0.36, '_': 0.38, '-': 0.4,
            '?': 0.42, ']': 0.44, '[': 0.46, '}': 0.48, '{': 0.5, '1': 0.52,
            ')': 0.54, '(': 0.56, '|': 0.58, '\\': 0.6, '/': 0.62, 't': 0.64,
            'f': 0.66, 'j': 0.68, 'r': 0.7, 'x': 0.72, 'n': 0.74, 'u': 0.76,
            'v': 0.78, 'c': 0.8, 'z': 0.82, 'X': 0.84, 'Y': 0.86, 'U': 0.88,
            'J': 0.9, 'C': 0.92, 'L': 0.94, 'Q': 0.96, '0': 0.98, 'O': 1.0,
            'm': 1.02, 'w': 1.04, 'q': 1.06, 'p': 1.08, 'd': 1.1, 'b': 1.12,
            'k': 1.14, 'h': 1.16, 'a': 1.18, 'o': 1.2, '*': 1.22, '#': 1.24,
            'M': 1.26, 'W': 1.28, '&': 1.3, '8': 1.32, '%': 1.34, 'B': 1.36,
            '@': 1.38, '$': 1.4,
            # Block characters with proper density values
            '█': 1.5, '▉': 1.4, '▊': 1.3, '▋': 1.2, '▌': 1.1, '▍': 1.0, '▎': 0.9, '▏': 0.8,
            '▓': 1.2, '▒': 0.8, '░': 0.4, '▄': 0.6, '▀': 0.6, '■': 1.3, '□': 0.3,
            '▪': 1.1, '▫': 0.2, '●': 1.2, '○': 0.3, '◐': 0.7, '◑': 0.7, '◒': 0.7, '◓': 0.7,
            '▁': 0.1, '▂': 0.2, '▃': 0.3, '▄': 0.4, '▅': 0.5, '▆': 0.6, '▇': 0.7,
            '▐': 0.5, '▌': 0.5, '▊': 0.7, '▋': 0.6
        }
        
        # Face detection (UNCHANGED)
        self.face_cascade = None
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            pass
        
        self.current_image = None
        self.ascii_art = ""
        self.colored_ascii_art = ""
    
    # ALL ORIGINAL PROFESSIONAL METHODS (UNCHANGED)
    def perceptual_grayscale_conversion(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale using perceptual weights for human vision"""
        if len(image.shape) == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
            return gray.astype(np.uint8)
        return image
    
    def apply_clahe_enhancement(self, image: np.ndarray, clip_limit: float = 3.0, 
                               tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply CLAHE for dramatic local detail enhancement"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    def apply_bilateral_filtering(self, image: np.ndarray, d: int = 9, 
                                 sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """Apply bilateral filtering to reduce noise while preserving edges"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def apply_unsharp_masking(self, image: np.ndarray, sigma: float = 1.5, 
                             strength: float = 1.5, threshold: float = 0) -> np.ndarray:
        """Apply unsharp masking for enhanced detail sharpness"""
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        unsharp_mask = cv2.addWeighted(image, 1 + strength, blurred, -strength, threshold)
        return np.clip(unsharp_mask, 0, 255).astype(np.uint8)
    
    def calibrate_character_set(self, character_set: str) -> List[Tuple[str, float]]:
        """Sort character set by actual visual density"""
        char_density_pairs = []
        
        for char in character_set:
            density = self.character_densities.get(char, 0.5)
            char_density_pairs.append((char, density))
        
        char_density_pairs.sort(key=lambda x: x[1])
        return char_density_pairs
    
    def correct_aspect_ratio(self, image: Image.Image, target_width: int, 
                           char_aspect_ratio: float = 0.55) -> Image.Image:
        """Correct aspect ratio for character dimensions"""
        original_width, original_height = image.size
        target_height = int(target_width * original_height / original_width * char_aspect_ratio)
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    def create_edge_map(self, image: np.ndarray, low_threshold: int = 50, 
                       high_threshold: int = 150, edge_strength: float = 0.15) -> np.ndarray:
        """Create edge map for feature enhancement"""
        edges = cv2.Canny(image, low_threshold, high_threshold)
        edge_map = (edges.astype(np.float32) / 255.0) * edge_strength
        return edge_map
    
    def advanced_character_mapping(self, pixel_value: int, edge_value: float, 
                                 calibrated_chars: List[Tuple[str, float]], 
                                 use_s_curve: bool = True) -> str:
        """Map pixel to character using calibrated densities"""
        if not calibrated_chars:
            return ' '
        
        normalized = pixel_value / 255.0
        enhanced = min(1.0, normalized + edge_value)
        
        if use_s_curve:
            if enhanced < 0.5:
                enhanced = 2 * enhanced * enhanced
            else:
                enhanced = 1 - 2 * (1 - enhanced) * (1 - enhanced)
        
        char_index = int(enhanced * (len(calibrated_chars) - 1))
        char_index = max(0, min(len(calibrated_chars) - 1, char_index))
        
        return calibrated_chars[char_index][0]


    # NEW METHODS FOR ENHANCED FUNCTIONALITY
    def apply_artistic_style(self, image: np.ndarray, style: str) -> np.ndarray:
        """Apply artistic style transformations"""
        if style == 'sketch':
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        elif style == 'painting':
            return cv2.bilateralFilter(image, 15, 80, 80)
        
        elif style == 'comic':
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            bilateral = cv2.bilateralFilter(image, 9, 200, 200)
            return cv2.bitwise_and(bilateral, edges)
        
        elif style == 'vintage':
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            return cv2.transform(image, kernel)
        
        elif style == 'cyberpunk':
            blur = cv2.GaussianBlur(image, (15, 15), 0)
            return cv2.addWeighted(image, 1.5, blur, -0.5, 0)
        
        elif style == 'minimalist':
            data = image.reshape((-1, 3))
            data = np.float32(data)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            return segmented_data.reshape(image.shape)
        
        elif style == 'abstract':
            rows, cols = image.shape[:2]
            src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
            dst_points = np.float32([[0, rows*0.1], [cols*0.9, 0], [cols*0.1, rows*0.9], [cols, rows*0.8]])
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            return cv2.warpPerspective(image, matrix, (cols, rows))
        
        return image
    
    def generate_color_map(self, image: np.ndarray, palette_name: str) -> np.ndarray:
        """Generate color mapping for colored ASCII - FIXED VERSION"""
        palette = self.color_palettes.get(palette_name, self.color_palettes['grayscale'])
        
        if len(image.shape) == 3:
            gray = self.perceptual_grayscale_conversion(image)
        else:
            gray = image
        
        color_mapped = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        
        # Improved color mapping with better distribution
        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                intensity = gray[y, x] / 255.0
                
                # Map intensity to palette index
                palette_index = int(intensity * (len(palette) - 1))
                palette_index = max(0, min(len(palette) - 1, palette_index))
                
                color_mapped[y, x] = palette[palette_index]
        
        return color_mapped
    
    def generate_ultimate_ascii(self, image_path: str, width: int = 120,
                               character_set: str = 'professional',
                               artistic_style: str = 'photorealistic',
                               color_palette: str = 'grayscale',
                               enable_color: bool = False,
                               clahe_clip_limit: float = 3.0,
                               bilateral_d: int = 9,
                               unsharp_strength: float = 1.5,
                               edge_enhancement: float = 0.15,
                               use_s_curve: bool = True,
                               crop_to_face: bool = True) -> Tuple[str, str]:
        """Generate ultimate ASCII art with all enhancements"""
        
        try:
            # Step 1: Load and prepare image (UNCHANGED)
            image = Image.open(image_path)
            self.current_image = image.copy()
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Step 2: Crop to face if requested (UNCHANGED)
            if crop_to_face and self.face_cascade is not None:
                gray_for_detection = self.perceptual_grayscale_conversion(img_array)
                faces = self.face_cascade.detectMultiScale(gray_for_detection, 1.1, 4)
                
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    
                    padding = int(min(w, h) * 0.3)
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(img_array.shape[1] - x, w + 2 * padding)
                    h = min(img_array.shape[0] - y, h + 2 * padding)
                    
                    img_array = img_array[y:y+h, x:x+w]
            
            # Step 3: Apply artistic style transformation (NEW)
            if artistic_style != 'photorealistic':
                img_array = self.apply_artistic_style(img_array, artistic_style)
            
            # Step 4: Convert to perceptual grayscale (UNCHANGED)
            gray_image = self.perceptual_grayscale_conversion(img_array)
            
            # Step 5: Apply professional enhancements (UNCHANGED)
            enhanced_image = self.apply_clahe_enhancement(gray_image, clahe_clip_limit)
            filtered_image = self.apply_bilateral_filtering(enhanced_image, bilateral_d)
            sharpened_image = self.apply_unsharp_masking(filtered_image, strength=unsharp_strength)
            
            # Step 6: Correct aspect ratio and resize (UNCHANGED)
            pil_image = Image.fromarray(sharpened_image)
            resized_image = self.correct_aspect_ratio(pil_image, width)
            final_array = np.array(resized_image)
            
            # Step 7: Create edge map (UNCHANGED)
            edge_map = self.create_edge_map(final_array, edge_strength=edge_enhancement)
            
            # Step 8: Generate color map if needed (NEW)
            color_map = None
            if enable_color:
                color_pil = Image.fromarray(img_array)
                color_resized = self.correct_aspect_ratio(color_pil, width)
                color_array = np.array(color_resized)
                color_map = self.generate_color_map(color_array, color_palette)
            
            # Step 9: Calibrate character set (ENHANCED)
            char_set = self.character_sets.get(character_set, self.calibrated_character_set)
            
            # All character sets now use proper calibration
            calibrated_chars = self.calibrate_character_set(char_set)
            
            # Step 10: Generate ASCII art (ENHANCED)
            ascii_lines = []
            height, width_px = final_array.shape
            
            for y in range(height):
                line = ""
                for x in range(width_px):
                    pixel_value = final_array[y, x]
                    edge_value = edge_map[y, x] if edge_enhancement > 0 else 0.0
                    
                    # Use consistent advanced character mapping for all sets
                    char = self.advanced_character_mapping(
                        pixel_value, edge_value, calibrated_chars, use_s_curve
                    )
                    
                    line += char
                ascii_lines.append(line)
            
            # Step 11: Create colored version if requested (NEW)
            colored_ascii = ""
            if enable_color and color_map is not None:
                colored_ascii = self.create_colored_ascii(ascii_lines, color_map)
                self.colored_ascii_art = colored_ascii
            
            self.ascii_art = '\n'.join(ascii_lines)
            return self.ascii_art, colored_ascii
            
        except Exception as e:
            raise Exception(f"Error generating ultimate ASCII art: {str(e)}")
    
    def create_colored_ascii(self, ascii_lines: List[str], color_map: np.ndarray) -> str:
        """Create colored ASCII - FIXED VERSION"""
        colored_lines = []
        
        for y, line in enumerate(ascii_lines):
            colored_line = ""
            for x, char in enumerate(line):
                if y < color_map.shape[0] and x < color_map.shape[1]:
                    r, g, b = color_map[y, x]
                    # Create HTML-style colored text for better display
                    colored_line += f"<span style='color: rgb({r},{g},{b})'>{char}</span>"
                else:
                    colored_line += char
            colored_lines.append(colored_line)
        
        return '\n'.join(colored_lines)
    
    def save_ascii_art(self, output_path: str, ascii_art: str = None) -> None:
        """Save ASCII art with UTF-8 encoding"""
        if ascii_art is None:
            ascii_art = self.ascii_art
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ascii_art)
    
    def export_as_image(self, output_path: str, font_size: int = 8, 
                       bg_color: str = 'black', text_color: str = 'white') -> None:
        """Export ASCII art as image with monospace font"""
        if not self.ascii_art:
            raise Exception("No ASCII art to export")
        
        lines = self.ascii_art.split('\n')
        if not lines:
            return
        
        max_width = max(len(line) for line in lines)
        height = len(lines)
        
        img_width = max_width * font_size
        img_height = height * font_size
        
        image = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("consola.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("courier.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        y_pos = 0
        for line in lines:
            draw.text((0, y_pos), line, fill=text_color, font=font)
            y_pos += font_size
        
        image.save(output_path)


class UltimateASCIIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultimate ASCII Art Generator - Professional + Color + Artistic Styles")
        self.root.geometry("1600x1000")
        
        self.generator = UltimateASCIIGenerator()
        self.current_image_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup ultimate UI with all features"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Ultimate controls
        control_frame = ttk.LabelFrame(main_frame, text="Ultimate Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # File selection
        ttk.Button(control_frame, text="Select Image", 
                  command=self.select_image).pack(fill=tk.X, pady=2)
        
        # Character set selection
        ttk.Label(control_frame, text="Character Set:").pack(anchor=tk.W, pady=(10, 2))
        self.char_set_var = tk.StringVar(value='professional')
        char_combo = ttk.Combobox(control_frame, textvariable=self.char_set_var,
                                 values=list(self.generator.character_sets.keys()),
                                 state="readonly")
        char_combo.pack(fill=tk.X, pady=2)
        char_combo.bind('<<ComboboxSelected>>', self.on_parameter_change)
        
        # Artistic Style Selection
        ttk.Label(control_frame, text="Artistic Style:").pack(anchor=tk.W, pady=(10, 2))
        self.style_var = tk.StringVar(value='photorealistic')
        style_combo = ttk.Combobox(control_frame, textvariable=self.style_var,
                                  values=['photorealistic', 'sketch', 'painting', 'comic', 'vintage', 'cyberpunk', 'minimalist', 'abstract'],
                                  state="readonly")
        style_combo.pack(fill=tk.X, pady=2)
        style_combo.bind('<<ComboboxSelected>>', self.on_parameter_change)
        
        # Color Options
        color_frame = ttk.LabelFrame(control_frame, text="Color Options", padding=5)
        color_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.enable_color_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(color_frame, text="Enable Color ASCII", 
                       variable=self.enable_color_var,
                       command=self.on_parameter_change).pack(anchor=tk.W, pady=2)
        
        ttk.Label(color_frame, text="Color Palette:").pack(anchor=tk.W, pady=(5, 2))
        self.palette_var = tk.StringVar(value='rainbow')
        palette_combo = ttk.Combobox(color_frame, textvariable=self.palette_var,
                                    values=list(self.generator.color_palettes.keys()),
                                    state="readonly")
        palette_combo.pack(fill=tk.X, pady=2)
        palette_combo.bind('<<ComboboxSelected>>', self.on_parameter_change)
        
        # Width control
        ttk.Label(control_frame, text="Width (100-200 recommended):").pack(anchor=tk.W, pady=(10, 2))
        self.width_var = tk.IntVar(value=120)
        width_scale = ttk.Scale(control_frame, from_=60, to=250, 
                               variable=self.width_var, orient=tk.HORIZONTAL,
                               command=self.on_parameter_change)
        width_scale.pack(fill=tk.X, pady=2)
        
        # Professional controls
        prof_frame = ttk.LabelFrame(control_frame, text="Professional Settings", padding=5)
        prof_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(prof_frame, text="CLAHE Enhancement:").pack(anchor=tk.W, pady=(5, 2))
        self.clahe_var = tk.DoubleVar(value=3.0)
        ttk.Scale(prof_frame, from_=1.0, to=8.0, variable=self.clahe_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X, pady=2)
        
        ttk.Label(prof_frame, text="Edge Enhancement:").pack(anchor=tk.W, pady=(5, 2))
        self.edge_var = tk.DoubleVar(value=0.15)
        ttk.Scale(prof_frame, from_=0.0, to=0.5, variable=self.edge_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X, pady=2)
        
        self.s_curve_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prof_frame, text="S-Curve Enhancement", 
                       variable=self.s_curve_var,
                       command=self.on_parameter_change).pack(anchor=tk.W, pady=2)
        
        self.crop_face_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prof_frame, text="Auto-Crop to Face", 
                       variable=self.crop_face_var,
                       command=self.on_parameter_change).pack(anchor=tk.W, pady=2)
        
        # Generate buttons
        ttk.Button(control_frame, text="Generate Ultimate ASCII", 
                  command=self.generate_art).pack(fill=tk.X, pady=(20, 2))
        
        ttk.Button(control_frame, text="Save ASCII Art", 
                  command=self.save_art).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Export as Image", 
                  command=self.export_image).pack(fill=tk.X, pady=2)
        
        # Right panel - Preview
        preview_frame = ttk.LabelFrame(main_frame, text="Ultimate Preview", padding=10)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Color status indicator
        self.color_status_var = tk.StringVar(value="Colors: Disabled")
        color_status_label = ttk.Label(preview_frame, textvariable=self.color_status_var, 
                                      font=('Arial', 8, 'bold'))
        color_status_label.pack(anchor=tk.W, pady=(0, 5))
        
        text_frame = ttk.Frame(preview_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_widget = tk.Text(text_frame, font=('Consolas', 6), 
                                  wrap=tk.NONE, bg='black', fg='white')
        
        v_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, 
                                   command=self.text_widget.yview)
        h_scrollbar = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, 
                                   command=self.text_widget.xview)
        
        self.text_widget.configure(yscrollcommand=v_scrollbar.set,
                                  xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ultimate ASCII Generator Ready - All Professional Features + Color + Artistic Styles")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_image(self):
        """Select image for ultimate processing"""
        file_path = filedialog.askopenfilename(
            title="Select Image for Ultimate ASCII Conversion",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.status_var.set(f"Image selected: {os.path.basename(file_path)}")
            self.generate_art()
    
    def on_parameter_change(self, event=None):
        """Handle parameter changes for real-time preview"""
        if self.current_image_path:
            threading.Thread(target=self.generate_art, daemon=True).start()
    
    def display_colored_ascii(self, ascii_art: str, palette_name: str):
        """Display ASCII art with colors in the text widget"""
        # Clear existing content
        self.text_widget.delete(1.0, tk.END)
        
        # Get the color palette
        palette = self.generator.color_palettes.get(palette_name, self.generator.color_palettes['grayscale'])
        
        lines = ascii_art.split('\n')
        
        # Configure color tags for each color in the palette
        for i, (r, g, b) in enumerate(palette):
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
            tag_name = f"color_{i}"
            self.text_widget.tag_configure(tag_name, foreground=color_hex)
        
        # Get character density mapping for better color assignment
        char_densities = self.generator.character_densities
        
        # Insert text with colors based on character density
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                # Get character density and map to color
                density = char_densities.get(char, 0.5)
                
                # Normalize density to 0-1 range
                normalized_density = min(1.0, max(0.0, density / 1.5))
                
                # Map to color palette index
                color_index = int(normalized_density * (len(palette) - 1))
                color_index = max(0, min(len(palette) - 1, color_index))
                
                tag_name = f"color_{color_index}"
                self.text_widget.insert(tk.END, char, tag_name)
            
            if y < len(lines) - 1:  # Don't add newline after last line
                self.text_widget.insert(tk.END, '\n')
    
    def generate_art(self):
        """Generate ultimate ASCII art with all features"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        try:
            self.status_var.set("Generating ultimate ASCII art...")
            self.root.update()
            
            # Get all parameters
            width = self.width_var.get()
            char_set = self.char_set_var.get()
            artistic_style = self.style_var.get()
            color_palette = self.palette_var.get()
            enable_color = self.enable_color_var.get()
            clahe_limit = self.clahe_var.get()
            edge_enhancement = self.edge_var.get()
            use_s_curve = self.s_curve_var.get()
            crop_to_face = self.crop_face_var.get()
            
            # Generate ultimate ASCII art
            ascii_art, colored_ascii = self.generator.generate_ultimate_ascii(
                self.current_image_path, width, char_set, artistic_style,
                color_palette, enable_color, clahe_limit, 9, 1.5,
                edge_enhancement, use_s_curve, crop_to_face
            )
            
            # Update preview with color support
            if enable_color:
                # Display colored ASCII with proper color rendering
                self.display_colored_ascii(ascii_art, color_palette)
                self.color_status_var.set(f"Colors: Enabled ({color_palette.upper()} palette)")
            else:
                # Display regular ASCII
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, ascii_art)
                self.color_status_var.set("Colors: Disabled")
            
            self.status_var.set("Ultimate ASCII art generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate ASCII art: {str(e)}")
            self.status_var.set("Error generating ASCII art")
    
    def save_art(self):
        """Save ASCII art as text file"""
        if not self.generator.ascii_art:
            messagebox.showwarning("Warning", "No ASCII art to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Ultimate ASCII Art",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.generator.save_ascii_art(file_path)
                messagebox.showinfo("Success", f"ASCII art saved to {file_path}")
                self.status_var.set(f"Saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def export_image(self):
        """Export ASCII art as image"""
        if not self.generator.ascii_art:
            messagebox.showwarning("Warning", "No ASCII art to export!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export ASCII Art as Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.generator.export_as_image(file_path)
                messagebox.showinfo("Success", f"ASCII art exported to {file_path}")
                self.status_var.set(f"Exported to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")


def main():
    """Main function for ultimate ASCII generator"""
    parser = argparse.ArgumentParser(description="Ultimate ASCII Art Generator")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--input", "-i", help="Input image path")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--width", "-w", type=int, default=120, help="Width in characters")
    parser.add_argument("--charset", "-c", default="professional", help="Character set")
    parser.add_argument("--style", "-s", default="photorealistic", help="Artistic style")
    parser.add_argument("--color", action="store_true", help="Enable color ASCII")
    parser.add_argument("--palette", "-p", default="rainbow", help="Color palette")
    
    args = parser.parse_args()
    
    if args.cli:
        if not args.input:
            print("Error: Input image path required for CLI mode")
            return
        
        generator = UltimateASCIIGenerator()
        try:
            print("Generating ultimate ASCII art...")
            ascii_art, colored_ascii = generator.generate_ultimate_ascii(
                args.input, args.width, args.charset, args.style,
                args.palette, args.color
            )
            
            if args.output:
                generator.save_ascii_art(args.output)
                print(f"Ultimate ASCII art saved to {args.output}")
                
                if args.color and colored_ascii:
                    color_output = args.output.replace('.txt', '_colored.txt')
                    with open(color_output, 'w', encoding='utf-8') as f:
                        f.write(colored_ascii)
                    print(f"Colored ASCII art saved to {color_output}")
            else:
                print(ascii_art)
                
        except Exception as e:
            print(f"Error: {e}")
    else:
        # GUI mode
        root = tk.Tk()
        app = UltimateASCIIGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()
