from PIL import Image

# =========================
# 输入输出路径
# =========================
input_path = "Graphical Abstract.tif"
output_path = "Graphical Abstract_1200dpi.tiff"

# =========================
# 图像处理（resize + padding）
# =========================
img = Image.open(input_path).convert("RGB")

target_size = 1200
w, h = img.size

# 等比例缩放
scale = min(target_size / w, target_size / h)
new_w = int(w * scale)
new_h = int(h * scale)

img_resized = img.resize((new_w, new_h), Image.LANCZOS)

# 创建白底正方形
new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))

# 居中粘贴
paste_x = (target_size - new_w) // 2
paste_y = (target_size - new_h) // 2
new_img.paste(img_resized, (paste_x, paste_y))

# 保存（写入300 dpi）
new_img.save(output_path, dpi=(300, 300))

print("Done! Saved as:", output_path)

# =========================
# 自动检查（关键）
# =========================
check_img = Image.open(output_path)

width, height = check_img.size
dpi = check_img.info.get("dpi", None)

print("\n=== Verification ===")
print(f"Dimensions: {width} × {height}")

if dpi:
    print(f"Resolution: {dpi[0]} × {dpi[1]} dpi")
else:
    print("Resolution: Not found ❗")

# 判定
size_ok = (width == 1200 and height == 1200)
dpi_ok = (dpi is not None and int(dpi[0]) == 300 and int(dpi[1]) == 300)

print("\nCheck result:")
print("Size OK:", size_ok)
print("DPI OK:", dpi_ok)

if size_ok and dpi_ok:
    print("✅ PASS (符合 Cell Graphical Abstract 要求)")
else:
    print("❌ FAIL (需要调整)")