# Eksperimen SML Panji

Repository eksperimen dataset Employee Attrition untuk submission MSML.

## Struktur

```text
Eksperimen_SML_Panji
├── .github/workflows/preprocessing.yml
├── employee_attrition_raw
│   ├── README.md
│   └── employee_attrition_raw.csv
└── preprocessing
    ├── Eksperimen_Panji.ipynb
    ├── automate_Panji.py
    ├── requirements.txt
    └── employee_attrition_preprocessing
```

## Notebook Eksperimen

Notebook `preprocessing/Eksperimen_Panji.ipynb` mengikuti template eksperimen MSML:

1. Perkenalan Dataset
2. Import Library
3. Memuat Dataset
4. Exploratory Data Analysis
5. Data Preprocessing

## Menjalankan Preprocessing Otomatis

```bash
cd preprocessing
pip install -r requirements.txt
python automate_Panji.py
```

Output akan tersimpan di:

```text
preprocessing/employee_attrition_preprocessing
```

## Workflow

Workflow `.github/workflows/preprocessing.yml` menjalankan `automate_Panji.py`, mengunggah artifact dataset preprocessing, dan melakukan commit ulang dataset preprocessing ketika ada perubahan.
