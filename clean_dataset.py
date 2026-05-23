import pandas as pd
from pathlib import Path
import re
import shutil

def clean_data():
    project_dir = Path("/Users/anika/Desktop/surgical_instrument_classifier")
    dataset_dir = project_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    csv_src = project_dir / "metadata.csv"
    csv_dst = dataset_dir / "metadata.csv"
    
    img_dir_src = project_dir / "processed"
    img_dir_dst = dataset_dir / "processed"
    
    # 1. Reorganize directories
    if img_dir_src.exists():
        if img_dir_dst.exists():
            shutil.rmtree(img_dir_dst)
        shutil.move(str(img_dir_src), str(img_dir_dst))
        print("Moved processed images directory to dataset/processed.")
        
    if csv_src.exists():
        shutil.move(str(csv_src), str(csv_dst))
        print("Moved metadata.csv to dataset/metadata.csv.")
        
    if not csv_dst.exists():
        print("Error: metadata.csv not found!")
        return

    # 2. Clean metadata
    df = pd.read_csv(csv_dst)
    
    # Clean category: remove logo and supplier words
    clean_cat_pattern = re.compile(
        r'\b(IndoSurgicals|medical|equipment|supplier|I\s+S|urgicals|pment|private|limited|rights|reserved|copyright|and)\b', 
        re.IGNORECASE
    )
    
    cleaned_categories = []
    for cat in df['category'].fillna('General'):
        cat_str = str(cat).replace('®', '').replace('®', '')
        cat_clean = clean_cat_pattern.sub('', cat_str)
        # Remove extra spaces and non-alphanumeric chars at ends
        cat_clean = re.sub(r'\s+', ' ', cat_clean).strip()
        cat_clean = cat_clean.strip('& ')
        # Capitalize words
        cat_clean = cat_clean.title()
        # Fallbacks
        if not cat_clean or cat_clean.lower() in ['and', '']:
            cat_clean = 'General Instruments'
        # Normalize category names
        if 'Forceps' in cat_clean and 'Handles' in cat_clean:
            cat_clean = 'Surgical Forceps & Scalpel Handles'
        elif 'Forceps' in cat_clean:
            cat_clean = 'Surgical Forceps'
        elif 'Scissors' in cat_clean:
            cat_clean = 'Surgical Scissors'
        elif 'Dissecting' in cat_clean:
            cat_clean = 'General Dissecting Forceps'
        elif 'Cardiothoracic' in cat_clean:
            cat_clean = 'Cardiothoracic Instruments'
        elif 'Cardio' in cat_clean and 'Spine' in cat_clean:
            cat_clean = 'Cardio & Spine Instruments'
        elif 'Spine' in cat_clean and 'Proctoscope' in cat_clean:
            cat_clean = 'Spine Instruments & Proctoscopes'
        elif 'Rectal' in cat_clean or 'Gyn' in cat_clean:
            if 'Kit' in cat_clean:
                cat_clean = 'GYN & Instrument Kits'
            else:
                cat_clean = 'GYN Instruments'
        elif 'Instrument Kit' in cat_clean or 'Surgery Set' in cat_clean or 'Lab Dissection' in cat_clean:
            cat_clean = 'Surgical Instrument Kits'
        elif 'Ent' in cat_clean:
            cat_clean = 'ENT Instruments'
        elif 'Ophthalmic' in cat_clean:
            cat_clean = 'Ophthalmic Instruments'
        elif 'Retractor' in cat_clean:
            cat_clean = 'Retractors'
        
        cleaned_categories.append(cat_clean)
        
    df['category'] = cleaned_categories
    
    # Clean names
    cleaned_names = []
    for name in df['name'].fillna(''):
        n = str(name).strip()
        n = n.replace('®', '')
        # Remove trailing SKU patterns
        n = re.sub(r'SKU:\s*[A-Za-z0-9\-/]+.*', '', n, flags=re.IGNORECASE)
        # Remove trailing size patterns if any
        n = re.sub(r'Size:\s*.*', '', n, flags=re.IGNORECASE)
        # Remove redundant spaces
        n = re.sub(r'\s+', ' ', n).strip()
        # Capitalize first letter of each word
        n = ' '.join([w.capitalize() if not w.startswith('(') else w for w in n.split()])
        
        # Minor fixes for cut-off names
        if n.startswith('S '):
            n = n[2:]
        elif n.startswith('S(curved)'):
            n = 'Forceps (Curved)'
        elif n.startswith('S (curved)'):
            n = 'Forceps (Curved)'
        elif n == '(Curved)' and str(df.iloc[len(cleaned_names)]['sku']).strip() == '93116':
            n = 'Artery Forceps (Curved)'
        elif n == '(Curved)' and str(df.iloc[len(cleaned_names)]['sku']).strip() == '93345':
            n = 'Crile Artery Forceps (Curved)'
            
        cleaned_names.append(n)
        
    df['name'] = cleaned_names
    
    # Drop rows where name is empty or just whitespace
    df = df[df['name'].str.strip().str.len() > 0]
    
    # Clean image paths
    df['image_path'] = df['image_path'].apply(lambda p: f"dataset/processed/{Path(p).name}")
    
    df.to_csv(csv_dst, index=False)
    print("Cleaned and formatted metadata.csv.")
    print("Dataset cleanup completed successfully!")

if __name__ == "__main__":
    clean_data()
