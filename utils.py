import pandas as pd
import tempfile
from pathlib import Path
from config import GLOBAL_MEMORY

def convert_excel_to_csv(excel_path, output_dir=None):
    """Convert Excel file (xlsx, xls) to CSV format."""
    try:
        excel_path = Path(excel_path)
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="excel_conversion_")
        
        print(f"📊 Converting Excel file: {excel_path.name}")
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
        
        # Read the first sheet
        df = pd.read_excel(excel_path, sheet_name=0)
        
        csv_filename = f"{excel_path.stem}_converted.csv"
        csv_path = Path(output_dir) / csv_filename
        df.to_csv(csv_path, index=False)
        
        GLOBAL_MEMORY["excel_info"] = {
            "original_file": str(excel_path),
            "sheet_names": sheet_names,
            "sheet_used": sheet_names[0],
            "total_sheets": len(sheet_names)
        }
        
        return str(csv_path), output_dir
        
    except Exception as e:
        print(f"❌ Excel conversion error: {e}")
        raise