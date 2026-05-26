import os
import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from src.inference import YOLOv11Inference


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="YOLOv11 Object Search",
    layout="wide"
)

st.title("🔍 Computer Vision Powered Search Application")


# ---------------- SESSION STATE ----------------

if "metadata" not in st.session_state:
    st.session_state.metadata = []


# ---------------- SIDEBAR ----------------

st.sidebar.header("Settings")

model_path = st.sidebar.text_input(
    "Model weights path",
    value="yolo11m.pt"
)


# ---------------- IMAGE UPLOAD ----------------

st.header("📤 Upload Images")

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# ---------------- START INFERENCE ----------------

if st.button("Start Inference"):

    if not uploaded_files:
        st.warning("Please upload at least one image.")

    else:

        # Create temporary folder
        temp_dir = "temp_images"

        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded files
        for uploaded_file in uploaded_files:

            save_path = os.path.join(
                temp_dir,
                uploaded_file.name
            )

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.info("Running YOLO inference...")

        # Initialize inference
        inference = YOLOv11Inference(model_path)

        # Process images
        metadata = inference.process_directory(temp_dir)

        # Save metadata in session
        st.session_state.metadata = metadata

        # Save metadata.json
        os.makedirs("processed", exist_ok=True)

        metadata_path = "processed/metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        st.success(
            f"Processed {len(metadata)} images successfully!"
        )

        st.success(
            f"Metadata saved to: {metadata_path}"
        )


# ---------------- SEARCH ENGINE ----------------

st.header("🔎 Search Objects")

metadata = st.session_state.metadata


if metadata:

    # Collect all detected classes
    all_classes = set()

    for item in metadata:

        for det in item["detections"]:

            all_classes.add(det["class"])

    all_classes = sorted(list(all_classes))

    # Search mode
    search_mode = st.radio(
        "Search mode",
        ["OR", "AND"],
        horizontal=True
    )

    # Select objects
    selected_classes = st.multiselect(
        "Select objects to search",
        all_classes
    )

    # Search button
    if st.button("Search Images"):

        matched_results = []

        for item in metadata:

            image_classes = [
                det["class"]
                for det in item["detections"]
            ]

            if search_mode == "OR":

                if any(
                    cls in image_classes
                    for cls in selected_classes
                ):
                    matched_results.append(item)

            else:

                if all(
                    cls in image_classes
                    for cls in selected_classes
                ):
                    matched_results.append(item)

        st.success(
            f"Found {len(matched_results)} matching images"
        )

        # ---------------- DISPLAY RESULTS ----------------

        for item in matched_results:

            st.subheader(
                os.path.basename(item["image_path"])
            )

            image = Image.open(item["image_path"])

            draw = ImageDraw.Draw(image)

            # Draw ONLY searched objects
            for det in item["detections"]:

                cls = det["class"]

                if cls not in selected_classes:
                    continue

                bbox = det["bbox"]

                conf = det["confidence"]

                draw.rectangle(
                    bbox,
                    outline="red",
                    width=3
                )

                draw.text(
                    (bbox[0], bbox[1]),
                    f"{cls} {conf:.2f}",
                    fill="red"
                )

            st.image(image, width=700)

            st.write("### Detected Search Objects")

            # Show ONLY searched objects
            for det in item["detections"]:

                if det["class"] not in selected_classes:
                    continue

                st.write(
                    f"• {det['class']} "
                    f"({det['confidence']:.2f})"
                )

            st.divider()

else:

    st.info(
        "Upload images and run inference first."
    )