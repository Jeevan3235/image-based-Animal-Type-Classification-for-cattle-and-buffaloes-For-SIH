import json
import csv
import pandas as pd
from datetime import datetime
import sqlite3
import os

class BPAIntegration:
    def __init__(self, database_path="animal_classification.db"):
        """
        Initialize BPA integration for auto-saving records
        """
        self.database_path = database_path
        self.init_database()
    
    def init_database(self):
        """
        Initialize SQLite database for storing classification records
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS animal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                animal_type TEXT,
                confidence REAL,
                body_length REAL,
                height_withers REAL,
                chest_width REAL,
                rump_angle REAL,
                body_condition_score REAL,
                image_path TEXT,
                filename TEXT,
                processed_by TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_record(self, classification_result, filename=None, processed_by="auto", notes=""):
        """
        Save classification record to database
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            measurements = classification_result.get('measurements', {})
            
            cursor.execute('''
                INSERT INTO animal_records (
                    timestamp, animal_type, confidence, body_length, height_withers,
                    chest_width, rump_angle, body_condition_score, image_path,
                    filename, processed_by, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                classification_result.get('timestamp', datetime.now().isoformat()),
                classification_result.get('animal_type', 'unknown'),
                classification_result.get('confidence', 0.0),
                measurements.get('body_length_pixels', 0),
                measurements.get('height_withers_pixels', 0),
                measurements.get('chest_width_pixels', 0),
                measurements.get('rump_angle_degrees', 0),
                measurements.get('body_condition_score', 0),
                classification_result.get('image_path', ''),
                filename or os.path.basename(classification_result.get('image_path', '')),
                processed_by,
                notes
            ))
            
            conn.commit()
            record_id = cursor.lastrowid
            conn.close()
            
            return record_id
            
        except Exception as e:
            print(f"Error saving record: {e}")
            return None
    
    def export_to_csv(self, output_path="animal_records_export.csv"):
        """
        Export records to CSV file
        """
        try:
            conn = sqlite3.connect(self.database_path)
            df = pd.read_sql_query("SELECT * FROM animal_records", conn)
            df.to_csv(output_path, index=False)
            conn.close()
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def get_records(self, limit=100, offset=0):
        """
        Retrieve records from database
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM animal_records 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            records = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            conn.close()
            
            # Convert to list of dictionaries
            result = []
            for record in records:
                result.append(dict(zip(columns, record)))
            
            return result
            
        except Exception as e:
            print(f"Error retrieving records: {e}")
            return []
    
    def generate_report(self, start_date=None, end_date=None):
        """
        Generate summary report of classifications
        """
        records = self.get_records(limit=1000)  # Get larger dataset for reporting
        
        if not records:
            return {"error": "No records found"}
        
        df = pd.DataFrame(records)
        
        # Filter by date if provided
        if start_date and end_date:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        report = {
            "total_records": len(df),
            "cattle_count": len(df[df['animal_type'] == 'cattle']),
            "buffalo_count": len(df[df['animal_type'] == 'buffalo']),
            "average_confidence": df['confidence'].mean(),
            "most_common_type": df['animal_type'].mode().iloc[0] if not df['animal_type'].mode().empty else "N/A",
            "date_range": {
                "start": df['timestamp'].min() if not df.empty else "N/A",
                "end": df['timestamp'].max() if not df.empty else "N/A"
            }
        }
        
        return report
