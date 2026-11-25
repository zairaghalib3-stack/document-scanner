import streamlit as st
import cv2
import numpy as np

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Professional Document Scanner",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Digital Image Processing - Document Scanner")

st.markdown("""
Upload your document image and convert it into a
clean, black-and-white scanned document.
""")

st.divider()


# ---------------- ORDER POINTS ----------------

def order_points(pts):
    pts = pts.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[1] = pts[np.argmin(d)]  # Top-right
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[3] = pts[np.argmax(d)]  # Bottom-left

    return rect


# ---------------- DOCUMENT PROCESSING ----------------

def process_document(img):

    if img is None or img.size == 0:
        return None

    orig = img.copy()

    height, width = img.shape[:2]

    if height < 50 or width < 50:
        return None

    # Resize without unnecessary enlargement
    scale = min(1.0, 1200 / max(height, width))

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    img_resized = cv2.resize(
        img,
        (new_width, new_height)
    )

    # Step 1: Grayscale
    gray = cv2.cvtColor(
        img_resized,
        cv2.COLOR_BGR2GRAY
    )

    # Blur
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Step 2: Edge Detection
    edges = cv2.Canny(
        blur,
        75,
        200
    )

    # Step 3: Contour Detection
    contours_result = cv2.findContours(
        edges.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnts = contours_result[-2]

    cnts = sorted(
        cnts,
        key=cv2.contourArea,
        reverse=True
    )[:10]

    doc = None

    for c in cnts:

        perimeter = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(
            c,
            0.02 * perimeter,
            True
        )

        if (
            len(approx) == 4
            and cv2.contourArea(approx) >
            0.10 * new_width * new_height
        ):
            doc = approx
            break

    if doc is None:
        return None

    # Draw detected document outline
    outline = img_resized.copy()

    cv2.drawContours(
        outline,
        [doc],
        -1,
        (0, 255, 0),
        3
    )

    # Convert resized coordinates to original coordinates
    scale_x = width / new_width
    scale_y = height / new_height

    original_points = doc.astype("float32")

    original_points[:, 0, 0] *= scale_x
    original_points[:, 0, 1] *= scale_y

    # Arrange four corners
    rect = order_points(original_points)

    # Step 4: Perspective Transformation
    output_width = 800
    output_height = 1100

    dst = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(
        rect,
        dst
    )

    warp = cv2.warpPerspective(
        orig,
        M,
        (output_width, output_height)
    )

    # Step 5: Adaptive Thresholding
    warp_gray = cv2.cvtColor(
        warp,
        cv2.COLOR_BGR2GRAY
    )

    warp_gray = cv2.medianBlur(
        warp_gray,
        3
    )

    scan = cv2.adaptiveThreshold(
        warp_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    return gray, edges, outline, warp, scan


# ---------------- IMAGE UPLOAD ----------------

uploaded_files = st.file_uploader(
    "📤 Upload Document Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# ---------------- DISPLAY RESULTS ----------------

if uploaded_files:

    for file in uploaded_files:

        st.divider()

        st.subheader(f"📄 Result: {file.name}")

        try:

            file_bytes = np.asarray(
                bytearray(file.read()),
                dtype=np.uint8
            )

            img = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            if img is None:
                st.error(
                    f"Cannot read image: {file.name}"
                )
                continue

            results = process_document(img)

            if results is None:

                st.error(
                    "Document ke 4 corners detect nahi hue. "
                    "Clear image aur proper background try karein."
                )

                st.image(
                    img,
                    channels="BGR",
                    caption="Original Image",
                    width=400
                )

                continue

            gray, edges, outline, warp, scan = results

            # Display five processing steps
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.image(
                    gray,
                    caption="1. Grayscale",
                    use_container_width=True
                )

            with col2:
                st.image(
                    edges,
                    caption="2. Edge Detection",
                    use_container_width=True
                )

            with col3:
                st.image(
                    outline,
                    channels="BGR",
                    caption="3. Contour Detection",
                    use_container_width=True
                )

            with col4:
                st.image(
                    warp,
                    channels="BGR",
                    caption="4. Perspective Transform",
                    use_container_width=True
                )

            with col5:
                st.image(
                    scan,
                    caption="5. Final B&W Scan",
                    use_container_width=True
                )

            # Download scanned document
            success, buffer = cv2.imencode(
                ".jpg",
                scan
            )

            if success:

                st.download_button(
                    label="⬇️ Download Scanned Document",
                    data=buffer.tobytes(),
                    file_name=f"scanned_{file.name.rsplit('.', 1)[0]}.jpg",
                    mime="image/jpeg",
                    key=f"download_{file.name}"
                )

                st.success(
                    "Document successfully processed!"
                )

        except Exception as e:

            st.error(
                f"Error processing {file.name}: {str(e)}"
            )


else:

    st.info(
        "Please upload one or more document images to start scanning."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Digital Image Processing Project | "
    "Developed using Python, OpenCV, NumPy and Streamlit"
)