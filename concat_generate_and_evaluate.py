#!/usr/bin/env python3
import os
import csv
import json
import logging
import random
import time
import torch
import torch.nn as nn
import numpy as np
import torchvision
import dlib
import itertools
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import clip
except ImportError:
    print("Warning: 'clip' module not found.")

# --- GLOBAL PATH CONFIGURATION ---
BASE_DIR = "metrics_output"
FAIRFACE_MODEL_DIR = os.path.join(BASE_DIR, "fair_face_model")
DLIB_MODEL_DIR = os.path.join(BASE_DIR, "dlib_models")

def setup_paths(metrics_base_dir):
    global BASE_DIR, FAIRFACE_MODEL_DIR, DLIB_MODEL_DIR
    BASE_DIR = os.path.abspath(metrics_base_dir)
    FAIRFACE_MODEL_DIR = os.path.join(BASE_DIR, "fair_face_model")
    DLIB_MODEL_DIR = os.path.join(BASE_DIR, "dlib_models")
    os.makedirs(BASE_DIR, exist_ok=True)

# Attribute Labels
race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
gender_labels = ['Male', 'Female']
age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
attributes_dict = {'race': race_labels, 'gender': gender_labels, 'age': age_labels}
combinations_to_analyze = [['race'], ['gender'], ['age'], ['race', 'gender'], ['race', 'age'], ['gender', 'age'], ['race', 'gender', 'age']]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)

def detect_face(image_paths, SAVE_DETECTED_AT):
    face_detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(os.path.join(DLIB_MODEL_DIR, "shape_predictor_5_face_landmarks.dat"))
    for image_path in image_paths:
        try:
            img = dlib.load_rgb_image(image_path)
            img = dlib.resize_image(img, rows=800, cols=800)
            dets = face_detector(img, 1)
            if len(dets) == 0: continue
            faces = dlib.full_object_detections()
            for rect in dets: faces.append(sp(img, rect))
            chips = dlib.get_face_chips(img, faces, size=300, padding=0.25)
            for idx, chip in enumerate(chips):
                base = os.path.splitext(os.path.basename(image_path))[0]
                dlib.save_image(chip, os.path.join(SAVE_DETECTED_AT, f"{base}_face{idx}.png"))
        except: continue

def predict_age_gender_race(save_prediction_at, imgs_path):
    img_names = [os.path.join(imgs_path, f) for f in os.listdir(imgs_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_names: return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(os.path.join(FAIRFACE_MODEL_DIR, "res34_fair_align_multi_7_20190809.pt"), map_location=device))
    model = model.to(device).eval()
    trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    res = []
    for img_p in img_names:
        try:
            img_raw = dlib.load_rgb_image(img_p)
            img_t = trans(img_raw).view(1, 3, 224, 224).to(device)
            out = model(img_t).cpu().detach().numpy().squeeze()
            res.append({'face_name_align': img_p, 'race': race_labels[np.argmax(softmax(out[:7]))], 'gender': gender_labels[np.argmax(softmax(out[7:9]))], 'age': age_labels[np.argmax(softmax(out[9:18]))]})
        except: continue
    pd.DataFrame(res).to_csv(save_prediction_at, index=False)

def calculate_combined_bias_metrics(prediction_csv_path, bias_w_csv, bias_p_csv):
    df = pd.read_csv(prediction_csv_path)
    df['original_image'] = df['face_name_align'].apply(lambda x: os.path.basename(x).split('_face')[0])
    bw_res, bp_res = [], []
    for group in combinations_to_analyze:
        na = np.prod([len(attributes_dict[a]) for a in group])
        combs = list(itertools.product(*(attributes_dict[a] for a in group))) if len(group) > 1 else attributes_dict[group[0]]
        
        # Bias-W
        fw = df.groupby(group).size() / len(df)
        bw_res.append({'Attribute_Group': "+".join(group), 'Bias-W': np.sqrt(sum((fw.get(c, 0) - (1/na))**2 for c in combs)/na)})
        
        # Bias-P
        for img, g_df in df.groupby('original_image'):
            fp = g_df.groupby(group).size() / len(g_df)
            bp_res.append({'Original_Image': img, 'Attribute_Group': "+".join(group), 'Bias-P': np.sqrt(sum((fp.get(c, 0) - (1/na))**2 for c in combs)/na)})
    pd.DataFrame(bw_res).to_csv(bias_w_csv, index=False)
    pd.DataFrame(bp_res).to_csv(bias_p_csv, index=False)

def calculate_ens_metrics(prediction_csv_path, output_csv):
    df = pd.read_csv(prediction_csv_path)
    res = []
    for group in combinations_to_analyze:
        p = df.groupby(group).size() / len(df)
        ens = np.exp(-sum(pi * np.log(pi) for pi in p.values if pi > 0))
        res.append({'Attribute': "+".join(group), 'ENS': ens})
    pd.DataFrame(res).to_csv(output_csv, index=False)

def calculate_kl_divergence(prediction_csv_path, output_csv):
    df = pd.read_csv(prediction_csv_path)
    res = []
    for group in combinations_to_analyze:
        labels = list(itertools.product(*(attributes_dict[a] for a in group))) if len(group) > 1 else attributes_dict[group[0]]
        p = df.groupby(group).size() / len(df)
        kl = sum(p.get(l, 1e-12) * np.log(p.get(l, 1e-12) / (1/len(labels))) for l in labels)
        res.append({'Attribute_Group': "+".join(group), 'KL_Divergence': kl})
    pd.DataFrame(res).to_csv(output_csv, index=False)

def calculate_icad_metrics(prediction_csv_path, output_csv):
    import clip
    df = pd.read_csv(prediction_csv_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    embs = {}
    for p in df['face_name_align'].unique():
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            embs[p] = model.encode_image(img).detach().cpu().numpy().flatten()
        except: continue
    res = []
    for group in combinations_to_analyze:
        df['tag'] = df[group].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        icads = []
        for _, g in df.groupby('tag'):
            vecs = [embs[p] for p in g['face_name_align'] if p in embs]
            if len(vecs) > 1:
                X = np.array(vecs)
                icads.append(np.linalg.norm(X - X.mean(axis=0), axis=1).mean())
        res.append({'Attribute': "+".join(group), 'Stratified_ICAD': np.mean(icads) if icads else 0.0})
    pd.DataFrame(res).to_csv(output_csv, index=False)

def process_paths_and_predict(img_paths, tag):
    det_dir = os.path.join(BASE_DIR, f"detected_faces_{tag}")
    ensure_dir(det_dir)
    out_csv = os.path.join(BASE_DIR, f"test_outputs_{tag}.csv")
    detect_face(img_paths, det_dir)
    predict_age_gender_race(out_csv, det_dir)
    calculate_combined_bias_metrics(out_csv, os.path.join(BASE_DIR, f"bias_w_metrics_{tag}.csv"), os.path.join(BASE_DIR, f"bias_p_metrics_{tag}.csv"))
    calculate_ens_metrics(out_csv, os.path.join(BASE_DIR, f"ens_{tag}.csv"))
    calculate_kl_divergence(out_csv, os.path.join(BASE_DIR, f"kl_divergence_metrics_{tag}.csv"))
    calculate_icad_metrics(out_csv, os.path.join(BASE_DIR, f"icad_metrics_{tag}.csv"))
