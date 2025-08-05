#!/usr/bin/env python3
"""
Professional Export Components for aNEOS Core

Provides comprehensive export capabilities including CSV, JSON, XML, and
structured data exports for NEO analysis results with academic formatting.
"""

import os
import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


class Exporter:
    """
    Professional data exporter for NEO analysis results.
    
    Provides structured export capabilities in multiple formats including
    CSV, JSON, XML, and Excel with academic-quality formatting.
    """
    
    def __init__(self, output_dir: str = "exports", logger: Optional[logging.Logger] = None):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory for export output
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        self.has_pandas = HAS_PANDAS
        self.has_openpyxl = HAS_OPENPYXL
        
        # Metadata for exports
        self.metadata = {
            "export_version": "1.0",
            "system": "aNEOS Professional Reporting System",
            "generated_by": "aNEOS Core Export Module",
            "format_standard": "Academic Research Compatible"
        }
    
    def add_export_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add metadata to export data.
        
        Args:
            data: Data dictionary to enhance
            
        Returns:
            Enhanced data with metadata
        """
        enhanced_data = {
            "metadata": {
                **self.metadata,
                "export_timestamp": datetime.now().isoformat(),
                "export_date": date.today().isoformat(),
                "data_count": len(data.get("neos", [])) if isinstance(data.get("neos"), list) else 0
            },
            **data
        }
        return enhanced_data
    
    def sanitize_data_for_export(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sanitize data for export by handling None values and data types.
        
        Args:
            data: List of NEO data dictionaries
            
        Returns:
            Sanitized data suitable for export
        """
        sanitized_data = []
        
        for neo in data:
            sanitized_neo = {}
            
            for key, value in neo.items():
                # Handle None values
                if value is None:
                    sanitized_neo[key] = ""
                # Handle boolean values
                elif isinstance(value, bool):
                    sanitized_neo[key] = str(value).lower()
                # Handle numeric values
                elif isinstance(value, (int, float)):
                    sanitized_neo[key] = value
                # Handle datetime objects
                elif isinstance(value, (datetime, date)):
                    sanitized_neo[key] = value.isoformat()
                # Handle lists and dictionaries
                elif isinstance(value, (list, dict)):
                    sanitized_neo[key] = json.dumps(value)
                # Handle everything else as string
                else:
                    sanitized_neo[key] = str(value)
            
            sanitized_data.append(sanitized_neo)
        
        return sanitized_data
    
    def export_to_csv(self, data: List[Dict[str, Any]], 
                     output_file: Optional[str] = None,
                     include_metadata: bool = True) -> str:
        """
        Export data to CSV format.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            include_metadata: Whether to include metadata
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"aneos_export_{timestamp}.csv"
        
        output_path = self.output_dir / output_file
        
        # Sanitize data
        sanitized_data = self.sanitize_data_for_export(data)
        
        if not sanitized_data:
            self.logger.warning("No data to export to CSV")
            return str(output_path)
        
        # Get all unique fieldnames
        fieldnames = set()
        for neo in sanitized_data:
            fieldnames.update(neo.keys())
        fieldnames = sorted(list(fieldnames))
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            if include_metadata:
                # Write metadata as comments
                csvfile.write(f"# Export generated by: {self.metadata['system']}\n")
                csvfile.write(f"# Generated on: {datetime.now().isoformat()}\n")
                csvfile.write(f"# Total records: {len(sanitized_data)}\n")
                csvfile.write(f"# Format version: {self.metadata['export_version']}\n")
                csvfile.write("#\n")
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sanitized_data)
        
        self.logger.info(f"CSV export saved to {output_path}")
        return str(output_path)
    
    def export_to_json(self, data: List[Dict[str, Any]], 
                      output_file: Optional[str] = None,
                      include_metadata: bool = True,
                      pretty_print: bool = True) -> str:
        """
        Export data to JSON format.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            include_metadata: Whether to include metadata
            pretty_print: Whether to format JSON for readability
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"aneos_export_{timestamp}.json"
        
        output_path = self.output_dir / output_file
        
        # Prepare export data
        export_data = {
            "neos": data,
            "summary": {
                "total_count": len(data),
                "anomalous_count": sum(1 for neo in data if (
                    neo.get("ai_validated_anomaly", False) or 
                    neo.get("category") == "ISO Candidate" or 
                    neo.get("delta_v_anomaly_score", 0) > 1.5
                )),
                "verified_count": sum(1 for neo in data if neo.get("anomaly_confidence", 0) > 10)
            }
        }
        
        if include_metadata:
            export_data = self.add_export_metadata(export_data)
        
        # Write JSON
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            if pretty_print:
                json.dump(export_data, jsonfile, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(export_data, jsonfile, ensure_ascii=False, default=str)
        
        self.logger.info(f"JSON export saved to {output_path}")
        return str(output_path)
    
    def export_to_xml(self, data: List[Dict[str, Any]], 
                     output_file: Optional[str] = None,
                     include_metadata: bool = True) -> str:
        """
        Export data to XML format.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            include_metadata: Whether to include metadata
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"aneos_export_{timestamp}.xml"
        
        output_path = self.output_dir / output_file
        
        # Create XML structure
        root = ET.Element("aneos_export")
        
        if include_metadata:
            metadata_elem = ET.SubElement(root, "metadata")
            for key, value in self.metadata.items():
                meta_elem = ET.SubElement(metadata_elem, key.replace(" ", "_"))
                meta_elem.text = str(value)
            
            # Add export timestamp
            timestamp_elem = ET.SubElement(metadata_elem, "export_timestamp")
            timestamp_elem.text = datetime.now().isoformat()
            
            # Add summary
            summary_elem = ET.SubElement(metadata_elem, "summary")
            summary_elem.set("total_count", str(len(data)))
            
            anomalous_count = sum(1 for neo in data if (
                neo.get("ai_validated_anomaly", False) or 
                neo.get("category") == "ISO Candidate" or 
                neo.get("delta_v_anomaly_score", 0) > 1.5
            ))
            summary_elem.set("anomalous_count", str(anomalous_count))
            
            verified_count = sum(1 for neo in data if neo.get("anomaly_confidence", 0) > 10)
            summary_elem.set("verified_count", str(verified_count))
        
        # Add NEO data
        neos_elem = ET.SubElement(root, "neos")
        
        for neo in data:
            neo_elem = ET.SubElement(neos_elem, "neo")
            
            for key, value in neo.items():
                # Clean key name for XML
                clean_key = key.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                field_elem = ET.SubElement(neo_elem, clean_key)
                
                if value is None:
                    field_elem.text = ""
                elif isinstance(value, bool):
                    field_elem.text = str(value).lower()
                elif isinstance(value, (list, dict)):
                    field_elem.text = json.dumps(value)
                else:
                    field_elem.text = str(value)
        
        # Write XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)  # Pretty print
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        self.logger.info(f"XML export saved to {output_path}")
        return str(output_path)
    
    def export_to_excel(self, data: List[Dict[str, Any]], 
                       output_file: Optional[str] = None,
                       include_metadata: bool = True) -> Optional[str]:
        """
        Export data to Excel format with multiple sheets.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            include_metadata: Whether to include metadata
            
        Returns:
            Path to exported file or None if failed
        """
        if not self.has_pandas or not self.has_openpyxl:
            self.logger.error("pandas and openpyxl required for Excel export")
            return None
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"aneos_export_{timestamp}.xlsx"
        
        output_path = self.output_dir / output_file
        
        # Sanitize data for DataFrame
        sanitized_data = self.sanitize_data_for_export(data)
        
        if not sanitized_data:
            self.logger.warning("No data to export to Excel")
            return str(output_path)
        
        # Create DataFrame
        df = pd.DataFrame(sanitized_data)
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='NEO_Data', index=False)
            
            # Summary sheet
            summary_data = []
            
            if include_metadata:
                summary_data.extend([
                    ["System", self.metadata['system']],
                    ["Export Version", self.metadata['export_version']],
                    ["Generated By", self.metadata['generated_by']],
                    ["Export Timestamp", datetime.now().isoformat()],
                    ["", ""],
                ])
            
            # Calculate statistics
            total_count = len(data)
            anomalous_count = sum(1 for neo in data if (
                neo.get("ai_validated_anomaly", False) or 
                neo.get("category") == "ISO Candidate" or 
                neo.get("delta_v_anomaly_score", 0) > 1.5
            ))
            verified_count = sum(1 for neo in data if neo.get("anomaly_confidence", 0) > 10)
            
            summary_data.extend([
                ["Total NEOs", total_count],
                ["Anomalous NEOs", anomalous_count],
                ["Verified Anomalies", verified_count],
                ["Unverified Anomalies", anomalous_count - verified_count],
                ["Normal NEOs", total_count - anomalous_count],
                ["", ""],
            ])
            
            # Category breakdown
            categories = {}
            for neo in data:
                category = neo.get("category") or neo.get("dynamic_category", "Uncategorized")
                categories[category] = categories.get(category, 0) + 1
            
            summary_data.append(["Category Breakdown", ""])
            for category, count in sorted(categories.items()):
                summary_data.append([category, count])
            
            # Create summary DataFrame and export
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # If there are anomalous NEOs, create a separate sheet
            anomalous_data = [
                neo for neo in data if (
                    neo.get("ai_validated_anomaly", False) or 
                    neo.get("category") == "ISO Candidate" or 
                    neo.get("delta_v_anomaly_score", 0) > 1.5
                )
            ]
            
            if anomalous_data:
                anomalous_sanitized = self.sanitize_data_for_export(anomalous_data)
                anomalous_df = pd.DataFrame(anomalous_sanitized)
                anomalous_df.to_excel(writer, sheet_name='Anomalous_NEOs', index=False)
            
            # Priority targets sheet (top 50)
            priority_data = sorted(data, key=lambda x: x.get("priority_score", 0), reverse=True)[:50]
            if priority_data and any(neo.get("priority_score", 0) > 0 for neo in priority_data):
                priority_sanitized = self.sanitize_data_for_export(priority_data)
                priority_df = pd.DataFrame(priority_sanitized)
                priority_df.to_excel(writer, sheet_name='Priority_Targets', index=False)
        
        self.logger.info(f"Excel export saved to {output_path}")
        return str(output_path)
    
    def export_summary_statistics(self, data: List[Dict[str, Any]], 
                                output_file: Optional[str] = None) -> str:
        """
        Export summary statistics as a structured text file.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"aneos_statistics_{timestamp}.txt"
        
        output_path = self.output_dir / output_file
        
        # Calculate comprehensive statistics
        total_count = len(data)
        
        # Anomaly statistics
        anomalous_data = [
            neo for neo in data if (
                neo.get("ai_validated_anomaly", False) or 
                neo.get("category") == "ISO Candidate" or 
                neo.get("delta_v_anomaly_score", 0) > 1.5
            )
        ]
        anomalous_count = len(anomalous_data)
        verified_count = sum(1 for neo in data if neo.get("anomaly_confidence", 0) > 10)
        
        # TAS statistics
        raw_tas_values = [neo.get("Raw TAS") for neo in data if neo.get("Raw TAS") not in [None, 0]]
        dynamic_tas_values = [neo.get("Dynamic TAS") for neo in data if neo.get("Dynamic TAS") not in [None, 0]]
        
        # Category breakdown
        categories = {}
        for neo in data:
            category = neo.get("category") or neo.get("dynamic_category", "Uncategorized")
            categories[category] = categories.get(category, 0) + 1
        
        # Priority statistics
        priority_values = [neo.get("priority_score", 0) for neo in data if neo.get("priority_score", 0) > 0]
        
        # Build statistics report
        lines = [
            "=" * 80,
            "aNEOS EXPORT STATISTICS SUMMARY",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"System: {self.metadata['system']}",
            "",
            "DATASET OVERVIEW:",
            "-" * 40,
            f"Total NEOs: {total_count}",
            f"Anomalous NEOs: {anomalous_count} ({anomalous_count/total_count*100:.1f}%)" if total_count > 0 else "Anomalous NEOs: 0",
            f"Verified Anomalies: {verified_count}",
            f"Unverified Anomalies: {anomalous_count - verified_count}",
            "",
            "TAS STATISTICS:",
            "-" * 40,
        ]
        
        if raw_tas_values:
            lines.extend([
                f"Raw TAS - Count: {len(raw_tas_values)}",
                f"Raw TAS - Mean: {sum(raw_tas_values)/len(raw_tas_values):.2f}",
                f"Raw TAS - Max: {max(raw_tas_values):.2f}",
                f"Raw TAS - Min: {min(raw_tas_values):.2f}",
            ])
        else:
            lines.append("Raw TAS - No data available")
        
        if dynamic_tas_values:
            lines.extend([
                f"Dynamic TAS - Count: {len(dynamic_tas_values)}",
                f"Dynamic TAS - Mean: {sum(dynamic_tas_values)/len(dynamic_tas_values):.2f}",
                f"Dynamic TAS - Max: {max(dynamic_tas_values):.2f}",
                f"Dynamic TAS - Min: {min(dynamic_tas_values):.2f}",
            ])
        else:
            lines.append("Dynamic TAS - No data available")
        
        lines.extend([
            "",
            "CATEGORY DISTRIBUTION:",
            "-" * 40,
        ])
        
        for category, count in sorted(categories.items()):
            percentage = count/total_count*100 if total_count > 0 else 0
            lines.append(f"{category}: {count} ({percentage:.1f}%)")
        
        if priority_values:
            lines.extend([
                "",
                "PRIORITY STATISTICS:",
                "-" * 40,
                f"NEOs with Priority Scores: {len(priority_values)}",
                f"Average Priority Score: {sum(priority_values)/len(priority_values):.2f}",
                f"Highest Priority Score: {max(priority_values):.2f}",
                f"Lowest Priority Score: {min(priority_values):.2f}",
            ])
        
        lines.extend([
            "",
            "=" * 80,
            f"End of Statistics Summary - {total_count} NEOs processed",
            "=" * 80
        ])
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"Statistics summary saved to {output_path}")
        return str(output_path)
    
    def export_all_formats(self, data: List[Dict[str, Any]], 
                          base_filename: Optional[str] = None) -> Dict[str, Optional[str]]:
        """
        Export data in all available formats.
        
        Args:
            data: List of NEO data dictionaries
            base_filename: Base filename for output files
            
        Returns:
            Dictionary mapping format names to file paths
        """
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"aneos_export_{timestamp}"
        
        exports = {}
        
        # Export in each format
        exports["csv"] = self.export_to_csv(data, f"{base_filename}.csv")
        exports["json"] = self.export_to_json(data, f"{base_filename}.json")
        exports["xml"] = self.export_to_xml(data, f"{base_filename}.xml")
        exports["statistics"] = self.export_summary_statistics(data, f"{base_filename}_stats.txt")
        
        # Excel export if available
        if self.has_pandas and self.has_openpyxl:
            exports["excel"] = self.export_to_excel(data, f"{base_filename}.xlsx")
        else:
            exports["excel"] = None
            self.logger.info("Excel export skipped - pandas/openpyxl not available")
        
        # Filter out None results
        successful_exports = {k: v for k, v in exports.items() if v is not None}
        
        self.logger.info(f"Exported data in {len(successful_exports)} formats with base filename: {base_filename}")
        return exports