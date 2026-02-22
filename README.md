1. Calibration Phase: 

    python src/main.py \
    "data=calibration_gender,num_images=100,extract=True,protect=[gender]" \
    "proc=base"
    
    python src/main.py \
    "data=calibration_race,num_images=100,extract=True,protect=[race]" \
    "proc=base"
    
    python src/main.py \
    "data=calibration_intersectional,num_images=100,extract=True,protect=[gender,race]" \
    "proc=base"

2. Image generation Phase using custom prompts:

    python src/main.py \
    "data=custom,num_images=100,seed=42" \
    "proc=base"
    
    python src/main.py \
    "data=custom,num_images=100,protect=[gender],remove,hdim=100,seed=42" \
    "proc=fpca"
    
    python src/main.py \
    "data=custom,num_images=100,protect=[race],remove,hdim=100,seed=42" \
    "proc=fpca"
    
    CUDA_VISIBLE_DEVICES=1 python src/main.py \
    "data=custom,num_images=100,protect=[gender,race],remove,hdim=100,seed=42" \
    "proc=fpca"

3. Download models for bias metrics calculation:
   
    mkdir -p metrics_output/fair_face_model
    mkdir -p metrics_output/dlib_models
    
    Download FairFace model:
    curl -L -o metrics_output/fair_face_model/res34_fair_align_multi_7_20190809.pt \
    https://huggingface.co/wmpscc/StyleGene/resolve/main/res34_fair_align_multi_7_20190809.pt
    
    Download dlib landmark model:
    curl -L -o metrics_output/dlib_models/shape_predictor_5_face_landmarks.dat \
    https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/shape_predictor_5_face_landmarks.dat


4. Calculate the metrics:
    conda install -c conda-forge dlib cmake
    python evaluate_all_results.py
    python print_results.py


