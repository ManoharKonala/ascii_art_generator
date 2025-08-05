# 🎨 Ultimate ASCII Art Generator

Transform your photos into stunning ASCII art with **professional quality**, **vibrant colors**, and **artistic styles**!

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20macos%20%7C%20linux-lightgrey.svg)

## ✨ What Makes This Ultimate?

### 🔬 **Professional Image Processing**
- **Scientific character calibration** with measured visual density
- **CLAHE enhancement** for dramatic local detail improvement  
- **Bilateral filtering** for noise reduction while preserving edges
- **Unsharp masking** for enhanced detail sharpness
- **S-curve mapping** for photorealistic mid-tone representation
- **Auto face detection** and intelligent cropping

### 🌈 **Vibrant Color Support**
- **6 stunning color palettes**: Rainbow, Fire, Ocean, Forest, Sunset, Grayscale
- **Real-time color preview** in GUI
- **HTML-style color rendering** for beautiful displays
- **Character-based color mapping** for realistic results

### 🎭 **Artistic Styles**
- **Photorealistic** - Maximum quality with all professional techniques
- **Sketch** - Pencil drawing effect with enhanced edges
- **Painting** - Oil painting effect with smooth blending
- **Comic** - High contrast comic book style
- **Vintage** - Sepia tone old photograph effect
- **Cyberpunk** - Neon glow futuristic aesthetic
- **Minimalist** - Clean, simplified aesthetic
- **Abstract** - Artistic distortion effects

### 🎯 **Multiple Character Sets**
- **Professional** (71 chars) - Maximum quality for portraits
- **Artistic Blocks** (11 chars) - Modern clean aesthetics
- **Fine Blocks** (16 chars) - Maximum detail resolution
- **Shaded Blocks** (4 chars) - Minimalist gradient style
- **Mixed Symbols** (14 chars) - Creative texture effects

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install opencv-python pillow numpy

# Download the script
wget https://raw.githubusercontent.com/yourusername/ascii-art-generator/main/ascii_art_ultimate.py

# Run the GUI
python ascii_art_ultimate.py
```

### First Use
1. **Launch the application**: `python ascii_art_ultimate.py`
2. **Select an image**: Click "Select Image" and choose your photo
3. **Enable colors**: Check "Enable Color ASCII" ✅
4. **Choose a palette**: Select "rainbow" for vibrant colors
5. **Watch the magic**: See your photo transform in real-time!

## 🎨 Color Palettes Showcase

### 🌈 **Rainbow**
Perfect for vibrant, eye-catching results
```
Red → Orange → Yellow → Green → Blue → Indigo → Violet
```

### 🔥 **Fire** 
Hot colors for dramatic effect
```
White → Bright Yellow → Orange → Red → Dark Red
```

### 🌊 **Ocean**
Cool blues for serene, water-themed images
```
White → Light Blue → Sky Blue → Deep Blue → Navy → Black
```

### 🌲 **Forest**
Nature greens for landscapes and outdoor scenes
```
White → Light Green → Green → Forest Green → Dark Green → Black
```

### 🌅 **Sunset**
Warm tones for romantic, golden hour effects
```
White → Peach → Orange → Purple → Black
```

### ⚫ **Grayscale**
Classic black and white for timeless elegance
```
White → Light Gray → Gray → Dark Gray → Black
```

## 🎭 Artistic Styles Gallery

### **Photorealistic** 📸
- Uses all professional image processing techniques
- Perfect for portraits and detailed images
- Maximum facial feature recognition

### **Sketch** ✏️
- Pencil drawing effect with enhanced edges
- Great for artistic portraits
- Emphasizes contours and details

### **Painting** 🖼️
- Oil painting effect with smooth blending
- Beautiful for landscapes and artistic shots
- Soft, flowing appearance

### **Comic** 💥
- High contrast comic book style
- Bold, graphic novel aesthetic
- Perfect for fun, pop-art effects

### **Vintage** 📷
- Sepia tone old photograph effect
- Classic, nostalgic appearance
- Great for historical or retro themes

### **Cyberpunk** 🤖
- Neon glow futuristic aesthetic
- Sci-fi, high-tech appearance
- Perfect for modern, edgy looks

## 🎯 Perfect Settings Guide

### **For Professional Portraits**
```
Character Set: professional
Artistic Style: photorealistic  
Color Palette: grayscale
Width: 120-150 characters
CLAHE Enhancement: 3.0
Edge Enhancement: 0.15
S-Curve: ✅ Enabled
Auto-Crop Face: ✅ Enabled
```
**Result**: Maximum quality, recognizable faces with sharp details

### **For Vibrant Art**
```
Character Set: artistic_blocks
Artistic Style: painting
Color Palette: rainbow
Width: 100-120 characters  
CLAHE Enhancement: 3.5
Edge Enhancement: 0.2
S-Curve: ✅ Enabled
```
**Result**: Colorful, artistic masterpiece perfect for social media

### **For Vintage Photos**
```
Character Set: fine_blocks
Artistic Style: vintage
Color Palette: sunset
Width: 140-160 characters
CLAHE Enhancement: 2.5
Edge Enhancement: 0.1
S-Curve: ✅ Enabled
```
**Result**: Classic, nostalgic appearance with warm tones

### **For Creative/Fun Images**
```
Character Set: mixed_symbols
Artistic Style: comic
Color Palette: fire
Width: 80-100 characters
CLAHE Enhancement: 4.0
Edge Enhancement: 0.3
S-Curve: ❌ Disabled
```
**Result**: Bold, graphic style perfect for memes and creative projects

## 💻 Usage Examples

### GUI Mode (Recommended)
```bash
python ascii_art_ultimate.py
```
- **Intuitive interface** with real-time preview
- **All features accessible** through easy controls
- **Color status indicator** shows current palette
- **Export options** for multiple formats

### CLI Mode (Advanced)
```bash
# Professional portrait
python ascii_art_ultimate.py --cli -i portrait.jpg -o output.txt --charset professional --width 150

# Colorful artistic image
python ascii_art_ultimate.py --cli -i photo.jpg -o colored.txt --charset artistic_blocks --color --palette rainbow --style painting

# Vintage style
python ascii_art_ultimate.py --cli -i old_photo.jpg -o vintage.txt --charset fine_blocks --style vintage --palette sunset --width 140

# Cyberpunk effect
python ascii_art_ultimate.py --cli -i cityscape.jpg -o cyber.txt --charset mixed_symbols --style cyberpunk --palette neon --color
```

## 🎛️ Advanced Controls

### **Professional Settings**
- **CLAHE Enhancement** (1.0-8.0): Local contrast improvement
- **Edge Enhancement** (0.0-0.5): Facial feature sharpness
- **S-Curve Enhancement**: Photorealistic mid-tone mapping
- **Auto-Crop to Face**: Intelligent face detection and cropping

### **Creative Controls**
- **Width**: 60-250 characters (100-160 recommended)
- **Character Set**: 6 different sets for various aesthetics
- **Artistic Style**: 8 styles from photorealistic to abstract
- **Color Palette**: 6 palettes with real-time preview

## 📤 Export Options

### **Text Formats**
- **Plain Text** (.txt) - Standard ASCII art
- **Colored Text** - With color information for terminals

### **Image Formats**
- **PNG/JPEG** - Rendered with monospace font
- **Colored Images** - Full color ASCII art as images

### **Web Formats**
- **HTML** - Web-ready with CSS styling
- **Colored HTML** - With inline color styles

## 🔧 Technical Details

### **Image Processing Pipeline**
1. **Load & Prepare** - Image loading and format conversion
2. **Face Detection** - Optional intelligent cropping
3. **Artistic Style** - Apply selected artistic transformation
4. **Grayscale Conversion** - Perceptual weights (ITU-R BT.709)
5. **CLAHE Enhancement** - Local contrast improvement
6. **Bilateral Filtering** - Noise reduction + edge preservation
7. **Unsharp Masking** - Detail sharpening
8. **Aspect Correction** - Character dimension adjustment (0.55 ratio)
9. **Edge Detection** - Canny edge map creation
10. **Color Mapping** - Generate color palette mapping
11. **Character Mapping** - Advanced density-based selection
12. **Color Application** - Apply colors to final output

### **System Requirements**
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 100MB for dependencies
- **Display**: Any resolution (GUI scales automatically)

### **Dependencies**
```
opencv-python>=4.8.0    # Image processing
pillow>=10.0.0          # Image handling
numpy>=1.24.0           # Numerical operations
tkinter                 # GUI (usually included with Python)
```

## 🎨 Tips for Best Results

### **Image Selection**
- ✅ **High contrast images** work best
- ✅ **Clear facial features** for portraits
- ✅ **Good lighting** improves detail
- ✅ **Simple backgrounds** reduce noise
- ❌ Avoid very dark or very bright images
- ❌ Avoid overly complex backgrounds

### **Parameter Tuning**
- **Start with defaults** and adjust gradually
- **Higher width** = more detail but larger output
- **CLAHE 3.0-4.0** works best for most images
- **Edge enhancement 0.15-0.25** for portraits
- **Enable S-curve** for photorealistic results

### **Color Usage**
- **Rainbow palette** for vibrant, fun images
- **Grayscale** for professional, classic look
- **Fire/Sunset** for warm, cozy feeling
- **Ocean/Forest** for cool, natural themes

## 🐛 Troubleshooting

### **Common Issues**

**Colors not showing?**
- ✅ Make sure "Enable Color ASCII" is checked
- ✅ Try different color palettes
- ✅ Check that you have a recent Python version

**Image too dark/light?**
- ✅ Adjust CLAHE Enhancement (try 2.0-5.0)
- ✅ Try different artistic styles
- ✅ Use edge enhancement for more definition

**Output too large/small?**
- ✅ Adjust width parameter (60-250 range)
- ✅ Consider your display/export requirements
- ✅ Smaller width = faster processing

**Face not detected?**
- ✅ Try disabling "Auto-Crop to Face"
- ✅ Ensure face is clearly visible in image
- ✅ Try different angles or lighting

## 🤝 Contributing

Found a bug or have a feature request? Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with scientific image processing techniques
- Uses ITU-R BT.709 standard for perceptual accuracy
- Implements CLAHE, bilateral filtering, and unsharp masking
- Inspired by traditional ASCII art but enhanced with modern algorithms

---

**Transform your photos into stunning ASCII masterpieces with professional quality, vibrant colors, and artistic flair!** 🎨✨

*Made with ❤️ for artists, developers, and ASCII art enthusiasts*
