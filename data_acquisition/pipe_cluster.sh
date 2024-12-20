#!/bin/bash
#SBATCH --partition=owners # Specify the GPU partition
#SBATCH --gpus=1
#SBATCH --array=0-1639:2%20
#SBATCH -C GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB   
#SBATCH --cpus-per-task=6  # Request CPU cores
#SBATCH --mem-per-cpu=8G
#SBATCH --tasks=1
#SBATCH --time=40:00:00
#SBATCH --job-name=small_cls
#SBATCH --output=cluster_logs/%A/%A_%a.out
#SBATCH --error=cluster_logs/%A/%A_%a.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


cd /scratch/users/atacelen/

ml purge
ml load python/3.12.1
ml load cuda/12.4.0
ml load ffmpeg
source transfer/envs/mast3r/bin/activate

for (( i=$SLURM_ARRAY_TASK_ID; i<$SLURM_ARRAY_TASK_ID+2; i++ )); do
    filename="HouseTourVideos/Videos/${i}_video.mp4"
    name=$(basename "$filename")
    name="${name%.mp4}"
    
    echo "Processing $name"

    if [ ! -f "$filename" ]; then
        echo "File $filename does not exist."
        continue
    fi

    if [ -f "Reconstructions3D/$name/${i}_pointcloud.ply" ]; then
        # Continue processing
        echo "File exists. Continuing..."
        continue
    elif [ -d "Reconstructions3D/$name" ]; then
        # Delete the directory if pointcloud.ply doesn't exist
        echo "Directory exists but pointcloud is missing. Deleting directory..."
        rm -r "Reconstructions3D/$name"
    fi


    mkdir -p Reconstructions3D/$name
    mkdir Reconstructions3D/$name/images
    ffmpeg -i $filename -vf "fps=15" Reconstructions3D/$name/images/frame_%04d.png

    cd transfer

    python3 keyframe_extraction.py --folder ../Reconstructions3D/$name
    python3 blip_vqa.py --folder ../Reconstructions3D/$name
    python3 resize_keyframes.py --folder ../Reconstructions3D/$name

    cd ..

    singularity exec pipe colmap feature_extractor --database_path Reconstructions3D/$name/database_vocab.db --image_path Reconstructions3D/$name/keyframes_resized/ --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 --SiftExtraction.use_gpu 0
    singularity exec --nv pipe colmap vocab_tree_matcher --database_path Reconstructions3D/$name/database_vocab.db --VocabTreeMatching.vocab_tree_path transfer/vocab_tree.bin --VocabTreeMatching.num_images 100 --VocabTreeMatching.num_nearest_neighbors 5 --SiftMatching.guided_matching 1
    
    cd transfer

    python3 database_matches_to_txt.py --folder ../Reconstructions3D/$name
    python3 write_keypoints_h5.py --folder ../Reconstructions3D/$name
    
    cd mast3r

    python3 match_keyframes_mast3r_vocab.py --folder ../../Reconstructions3D/$name

    cd ..

    python3 colmap/h5_to_db.py --single-camera --database-path ../Reconstructions3D/$name/database.db ../Reconstructions3D/$name/ ../Reconstructions3D/$name/keyframes_resized/

    cd ..

    singularity exec --nv pipe colmap exhaustive_matcher --database_path Reconstructions3D/$name/database.db --SiftMatching.guided_matching 1
    singularity exec --nv pipe colmap mapper --database_path Reconstructions3D/$name/database.db --image_path Reconstructions3D/$name/keyframes_resized/ --output_path Reconstructions3D/$name --Mapper.ba_refine_focal_length 1 --Mapper.num_threads 6

    cd transfer/dust3r

    python3 run.py --folder /scratch/users/atacelen/Reconstructions3D/$name 

    cd ../..

    mv transfer/dust3r/${i}_pointcloud.ply Reconstructions3D/$name
    mv transfer/dust3r/pts3d.npy Reconstructions3D/$name
    # mv keyframes_resized/ Reconstructions3D/$name
    # mv transfer/database.db Reconstructions3D/$name
    # mv transfer/database_vocab.db Reconstructions3D/$name

    # rm -r transfer/matches.h5
    # rm -r transfer/keypoints.h5
    # rm -r transfer/vocab_matches.txt
    # rm -rf images/
    # rm -rf keyframes/

done
