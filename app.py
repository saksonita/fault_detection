import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import logging
import traceback
import psutil
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("industrial_fault_analysis")

# Configuration parameters with robust defaults
CONFIG = {
    "pressure_columns": ['[D0006894] 구동부측 역압', '[D0006896] 작업자측 역압'],
    "tension_columns": ['[D0006814] 언와인더 장력 PV', '[D0006816] 강제연신기 장력 PV', '[D0006818] 리와인더 장력 PV'],
    "gap_columns": ['[D0006860] 구동부측 GAP', '[D0006862] 작업자측 GAP', '[D0006864] 구동부측 이전 GAP', '[D0006866] 작업자측 이전 GAP'],
    "dancer_columns": ['[D0006926] 언와인더 댄서 위치.', '[D0006934] 리와인더 댄서 위치.', '[D0006936] 연신기 댄서 위치.'],
    "speed_column": '[D0006810] 프레스 속도 PV',
    "fault_column": '파단직전',
    "error_index_column": 'error_index',
    "timestamp_column": 'ds',
    "chunk_size": 100000,  # Chunk size for loading large files
    "memory_threshold": 0.75,  # Memory usage threshold (fraction of available memory)
    "max_points_visualization": 10000,  # Maximum points to render in visualizations
    "sampling_method": "systematic"  # Sampling method for large datasets: systematic, random, or none
}

# Column mapping for better readability
COLUMN_MAPPING = {
    '[D0006810] 프레스 속도 PV': 'Press Speed',
    '[D0006814] 언와인더 장력 PV': 'Unwinder Tension',
    '[D0006816] 강제연신기 장력 PV': 'Stretcher Tension',
    '[D0006818] 리와인더 장력 PV': 'Rewinder Tension',
    '[D0006860] 구동부측 GAP': 'Drive GAP',
    '[D0006862] 작업자측 GAP': 'Operator GAP',
    '[D0006864] 구동부측 이전 GAP': 'Previous Drive GAP',
    '[D0006866] 작업자측 이전 GAP': 'Previous Operator GAP',
    '[D0006894] 구동부측 역압': 'Drive Backpressure',
    '[D0006896] 작업자측 역압': 'Operator Backpressure',
    '[D0006926] 언와인더 댄서 위치.': 'Unwinder Dancer Position',
    '[D0006934] 리와인더 댄서 위치.': 'Rewinder Dancer Position',
    '[D0006936] 연신기 댄서 위치.': 'Stretcher Dancer Position',
    '프레스속도차이': 'Press Speed Diff',
    '언와인더장력차이': 'Unwinder Tension Diff',
    '리와인더장력차이': 'Rewinder Tension Diff',
    '강제연신기장력차이': 'Stretcher Tension Diff',
    '유도가열온도DS차이': 'Induction Temp DS Diff',
    '유도가열온도WS차이': 'Induction Temp WS Diff',
    '파단직전': 'Pre-Failure'
}

# Utility functions for memory optimization and data handling
def estimate_memory_usage(df):
    """
    Estimate the memory usage of a DataFrame in MB
    """
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    return memory_usage

def downsample_dataframe(df, max_points=10000, method='systematic'):
    """
    Downsample a DataFrame to a manageable size for visualization
    using different sampling methods.
    """
    if len(df) <= max_points:
        return df
    
    if method == 'random':
        # Random sampling
        return df.sample(max_points)
    elif method == 'systematic':
        # Systematic sampling (every nth row)
        n = len(df) // max_points
        return df.iloc[::n].head(max_points)
    else:
        # No downsampling
        return df

def validate_config(df, config):
    """
    Validate that required columns exist in the DataFrame
    """
    missing_columns = []
    
    # Critical columns
    critical_columns = [
        ("timestamp_column", config["timestamp_column"]),
        ("error_index_column", config["error_index_column"]),
        ("fault_column", config["fault_column"])
    ]
    
    for key, column in critical_columns:
        if column not in df.columns:
            missing_columns.append(f"{key}: {column}")
    
    # Warning for optional columns
    optional_column_groups = [
        ("pressure_columns", config["pressure_columns"]),
        ("tension_columns", config["tension_columns"]),
        ("gap_columns", config["gap_columns"]),
        ("dancer_columns", config["dancer_columns"]),
        ("speed_column", [config["speed_column"]])
    ]
    
    missing_optional = []
    for group_name, columns in optional_column_groups:
        missing_in_group = [col for col in columns if col not in df.columns]
        if missing_in_group:
            missing_optional.append(f"{group_name}: {', '.join(missing_in_group)}")
    
    return {
        "is_valid": len(missing_columns) == 0,
        "missing_critical": missing_columns,
        "missing_optional": missing_optional
    }

def ensure_column_types(df, config):
    """
    Ensure proper column data types for critical columns
    """
    try:
        # Ensure timestamp column is datetime
        if config["timestamp_column"] in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[config["timestamp_column"]]):
                df[config["timestamp_column"]] = pd.to_datetime(
                    df[config["timestamp_column"]], 
                    errors='coerce'
                )
                
        # Ensure error index is numeric
        if config["error_index_column"] in df.columns:
            if not pd.api.types.is_numeric_dtype(df[config["error_index_column"]]):
                df[config["error_index_column"]] = pd.to_numeric(
                    df[config["error_index_column"]], 
                    errors='coerce'
                )
        
        # Ensure fault column is numeric
        if config["fault_column"] in df.columns:
            if not pd.api.types.is_numeric_dtype(df[config["fault_column"]]):
                df[config["fault_column"]] = pd.to_numeric(
                    df[config["fault_column"]], 
                    errors='coerce'
                )
        
        return df, True, "Column types validated and converted successfully"
    
    except Exception as e:
        logger.error(f"Error ensuring column types: {str(e)}")
        return df, False, f"Error in type conversion: {str(e)}"

def load_data(file_path, config=CONFIG, chunk_size=None):
    """
    Load and preprocess the industrial fault data from CSV with optimized
    memory usage and chunked processing for large files.
    """
    if chunk_size is None:
        chunk_size = config["chunk_size"]
    
    try:
        start_time = time.time()
        logger.info(f"Starting to load data from {file_path}")
        
        # Check file size to determine loading strategy
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 100:  # For files larger than 100 MB
            logger.info(f"Large file detected. Using chunked loading with chunk size: {chunk_size}")
            
            # First, read a small sample to inspect columns and data types
            sample_df = pd.read_csv(file_path, nrows=1000)
            
            # Validate configuration with sample
            validation_result = validate_config(sample_df, config)
            if not validation_result["is_valid"]:
                missing_cols = ", ".join(validation_result["missing_critical"])
                return None, False, f"Critical columns missing: {missing_cols}"
            
            # Determine optimal data types to reduce memory usage
            dtypes = {}
            for col in sample_df.columns:
                if col == config["timestamp_column"]:
                    continue  # Will be parsed as datetime
                elif col == config["error_index_column"] or col == config["fault_column"]:
                    dtypes[col] = 'int32'
                elif pd.api.types.is_numeric_dtype(sample_df[col]):
                    # Use float32 instead of float64 to save memory
                    dtypes[col] = 'float32'
            
            # Read the file in chunks
            chunks = []
            for chunk in pd.read_csv(
                file_path, 
                dtype=dtypes,
                parse_dates=[config["timestamp_column"]] if config["timestamp_column"] in sample_df.columns else None,
                chunksize=chunk_size
            ):
                # Ensure proper column types
                chunk, _, _ = ensure_column_types(chunk, config)
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            
        else:
            # For smaller files, load all at once
            logger.info("Loading entire file at once")
            df = pd.read_csv(
                file_path, 
                parse_dates=[config["timestamp_column"]]
            )
            
            # Ensure proper column types
            df, type_conversion_success, type_conversion_message = ensure_column_types(df, config)
            if not type_conversion_success:
                logger.warning(type_conversion_message)
        
        # Validate configuration with loaded data
        validation_result = validate_config(df, config)
        if not validation_result["is_valid"]:
            missing_cols = ", ".join(validation_result["missing_critical"])
            return None, False, f"Critical columns missing: {missing_cols}"
        
        # Log optional missing columns as warnings
        if validation_result["missing_optional"]:
            missing_opt = ", ".join(validation_result["missing_optional"])
            logger.warning(f"Optional columns missing: {missing_opt}")
        
        # Basic data cleaning
        df = df.dropna(subset=[
            config["timestamp_column"], 
            config["error_index_column"], 
            config["fault_column"]
        ])
        
        # Log memory usage
        memory_usage = estimate_memory_usage(df)
        logger.info(f"DataFrame loaded. Shape: {df.shape}, Memory usage: {memory_usage:.2f} MB")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Data loading completed in {elapsed_time:.2f} seconds")
        
        return df, True, f"Data loaded successfully. {len(df)} rows, {len(df.columns)} columns."
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        return None, False, f"Error loading data: {str(e)}"

def process_error_index(df, error_index, config=CONFIG):
    """
    Process data for a specific error index with robust error handling
    """
    try:
        logger.info(f"Processing error index: {error_index}")
        
        # Filter data for the selected error index
        df_filtered = df[df[config["error_index_column"]] == error_index].copy()
        
        if df_filtered.empty:
            logger.warning(f"No data found for error index {error_index}")
            return df_filtered, False, None, f"No data found for error index {error_index}"
        
        # Verify timestamp column exists and is properly formatted
        if config["timestamp_column"] not in df_filtered.columns:
            logger.error(f"Timestamp column '{config['timestamp_column']}' not found")
            return df_filtered, False, None, f"Timestamp column '{config['timestamp_column']}' not found"
        
        # Convert timestamp to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_filtered[config["timestamp_column"]]):
            try:
                logger.info("Converting timestamp column to datetime")
                df_filtered[config["timestamp_column"]] = pd.to_datetime(df_filtered[config["timestamp_column"]])
            except Exception as e:
                logger.error(f"Failed to convert timestamp: {str(e)}")
                return df_filtered, False, None, f"Failed to convert timestamp: {str(e)}"
        
        # Sort by timestamp
        df_filtered = df_filtered.sort_values(by=config["timestamp_column"])
        
        # Look for fault condition with proper error handling
        if config["fault_column"] not in df_filtered.columns:
            logger.error(f"Fault column '{config['fault_column']}' not found")
            return df_filtered, False, None, f"Fault column '{config['fault_column']}' not found"
            
        fault_rows = df_filtered[df_filtered[config["fault_column"]] == 1]
        
        if len(fault_rows) > 0:
            # Get the first fault time
            fault_time = fault_rows[config["timestamp_column"]].iloc[0]
            logger.info(f"Fault found at {fault_time}")
            
            # Calculate time to fault in seconds with error handling
            try:
                df_filtered['seconds_to_fault'] = (df_filtered[config["timestamp_column"]] - fault_time).dt.total_seconds()
                logger.info(f"Time to fault calculated. Range: {df_filtered['seconds_to_fault'].min():.2f} to {df_filtered['seconds_to_fault'].max():.2f} seconds")
                return df_filtered, True, fault_time, f"Fault found at {fault_time}"
            except Exception as e:
                logger.error(f"Error calculating time to fault: {str(e)}")
                return df_filtered, False, None, f"Error calculating time to fault: {str(e)}"
        else:
            logger.warning(f"No fault condition detected for error index {error_index}")
            return df_filtered, False, None, f"No fault condition detected for error index {error_index}"
    
    except Exception as e:
        logger.error(f"Error processing error index {error_index}: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(), False, None, f"Processing error: {str(e)}"

def filter_by_time_window(df, time_window_seconds, config=CONFIG):
    """
    Filter data by time window around fault with error handling
    """
    try:
        min_time, max_time = time_window_seconds
        logger.info(f"Filtering by time window: {min_time} to {max_time} seconds")
        
        if 'seconds_to_fault' not in df.columns:
            logger.warning("Data does not contain time to fault calculation")
            return df, "Data does not contain time to fault calculation"
        
        filtered_df = df[(df['seconds_to_fault'] >= min_time) & (df['seconds_to_fault'] <= max_time)]
        
        if filtered_df.empty:
            logger.warning(f"No data found in the selected time window: {min_time} to {max_time} seconds")
            return df, f"No data found in the selected time window: {min_time} to {max_time} seconds"
        
        # Downsample if dataset is too large for efficient visualization
        if len(filtered_df) > config["max_points_visualization"]:
            original_size = len(filtered_df)
            filtered_df = downsample_dataframe(
                filtered_df, 
                config["max_points_visualization"], 
                config["sampling_method"]
            )
            logger.info(f"Downsampled from {original_size} to {len(filtered_df)} points for visualization")
        
        logger.info(f"Filtered to time window: {min_time} to {max_time} seconds. Resulting rows: {len(filtered_df)}")
        return filtered_df, f"Filtered to time window: {min_time} to {max_time} seconds. Results: {len(filtered_df)} rows"
    
    except Exception as e:
        logger.error(f"Error filtering by time window: {str(e)}")
        return df, f"Error filtering time window: {str(e)}"

def calculate_anomaly_scores(df, methods='zscore', config=CONFIG):
    """
    Calculate anomaly scores using various methods with error handling
    """
    try:
        logger.info(f"Calculating anomaly scores using method: {methods}")
        
        # Columns to use for anomaly detection
        numerical_columns = []
        for col_group in ["pressure_columns", "tension_columns", "gap_columns"]:
            numerical_columns.extend([col for col in config[col_group] if col in df.columns])
        
        if config["speed_column"] in df.columns:
            numerical_columns.append(config["speed_column"])
        
        # If no columns from configuration are found, use any numeric columns
        if not numerical_columns:
            logger.warning("No configured numerical columns found. Using available numeric columns.")
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            # Exclude certain columns that aren't relevant for anomaly detection
            exclude_cols = ['error_index', 'data_index', '파단직전', 'seconds_to_fault']
            numerical_columns = [col for col in numerical_columns if col not in exclude_cols]
            
        if not numerical_columns:
            logger.warning("No numerical columns found for anomaly detection")
            return df, "No numerical columns found for anomaly detection"
        
        # Create a working copy
        df_processed = df.copy()
        
        if methods == 'zscore':
            # Calculate z-scores for each numerical column
            for col in numerical_columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:  # Avoid division by zero
                    df_processed[f'{col}_zscore'] = (df[col] - mean_val) / std_val
            
            # Compute composite anomaly score as mean of absolute z-scores
            zscore_cols = [col for col in df_processed.columns if col.endswith('_zscore')]
            if zscore_cols:
                df_processed['anomaly_score'] = df_processed[zscore_cols].abs().mean(axis=1)
                logger.info(f"Anomaly scores calculated. Range: {df_processed['anomaly_score'].min():.4f} to {df_processed['anomaly_score'].max():.4f}")
            else:
                logger.warning("No z-score columns were calculated")
        
        elif methods == 'pca':
            # Use PCA for anomaly detection if we have enough data
            if len(df) > len(numerical_columns) and len(numerical_columns) > 1:
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numerical_columns])
                
                # Apply PCA
                n_components = min(len(numerical_columns), 3)
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(scaled_data)
                
                # Reconstruction error as anomaly score
                reconstructed = pca.inverse_transform(pca_result)
                reconstruction_error = np.sum((scaled_data - reconstructed) ** 2, axis=1)
                
                df_processed['anomaly_score'] = reconstruction_error
                logger.info(f"PCA anomaly scores calculated. Explained variance: {sum(pca.explained_variance_ratio_):.4f}")
                logger.info(f"Anomaly score range: {df_processed['anomaly_score'].min():.4f} to {df_processed['anomaly_score'].max():.4f}")
            else:
                logger.warning("Insufficient data for PCA-based anomaly detection")
                return df, "Insufficient data for PCA-based anomaly detection"
        
        return df_processed, f"Anomaly scores calculated using {methods} method"
    
    except Exception as e:
        logger.error(f"Error calculating anomaly scores: {str(e)}")
        logger.error(traceback.format_exc())
        return df, f"Error calculating anomaly scores: {str(e)}"

def plot_key_metrics(df, metric_type='tension', time_window_mins=[-5, 5], show_anomaly=True, config=CONFIG):
    """
    Generate plots for key metrics based on the selected type with error handling
    """
    try:
        if df is None or df.empty:
            logger.warning("No data available for plotting")
            return go.Figure().update_layout(title="No data available for plotting")
        
        # Get available columns in the dataframe
        available_columns = df.columns.tolist()
        logger.info(f"Available columns for plotting: {available_columns[:10]}... (total: {len(available_columns)})")
        
        # Initialize empty list for columns to plot
        columns_to_plot = []
        
        # Get the columns to plot based on the metric type
        if metric_type == 'tension':
            # First try exact column names from config
            columns_to_plot = [col for col in config["tension_columns"] if col in available_columns]
            # If no matches, try partial matches
            if not columns_to_plot:
                columns_to_plot = [col for col in available_columns if '장력' in col]
            title = "Tension Parameters"
            y_axis_title = "Tension"
            
        elif metric_type == 'pressure':
            columns_to_plot = [col for col in config["pressure_columns"] if col in available_columns]
            if not columns_to_plot:
                columns_to_plot = [col for col in available_columns if '역압' in col]
            title = "Pressure Parameters"
            y_axis_title = "Pressure"
            
        elif metric_type == 'gap':
            columns_to_plot = [col for col in config["gap_columns"] if col in available_columns]
            if not columns_to_plot:
                columns_to_plot = [col for col in available_columns if 'GAP' in col]
            title = "GAP Parameters"
            y_axis_title = "GAP"
            
        elif metric_type == 'dancer':
            columns_to_plot = [col for col in config["dancer_columns"] if col in available_columns]
            if not columns_to_plot:
                columns_to_plot = [col for col in available_columns if '댄서' in col]
            title = "Dancer Position Parameters"
            y_axis_title = "Position"
            
        elif metric_type == 'speed':
            if config["speed_column"] in available_columns:
                columns_to_plot = [config["speed_column"]]
            else:
                columns_to_plot = [col for col in available_columns if '속도' in col]
            title = "Press Speed"
            y_axis_title = "Speed"
            
        elif metric_type == 'diff':
            diff_params = ['프레스속도차이', '언와인더장력차이', '리와인더장력차이', '강제연신기장력차이']
            columns_to_plot = [col for col in diff_params if col in available_columns]
            if not columns_to_plot:
                columns_to_plot = [col for col in available_columns if '차이' in col]
            title = "Differential Parameters"
            y_axis_title = "Difference"
            
        elif metric_type == 'all':
            # Select numeric columns for visualization, excluding certain metadata columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            exclude_cols = ['error_index', 'data_index', 'anomaly_score', '파단직전', 'seconds_to_fault']
            exclude_cols.extend([col for col in numeric_cols if col.endswith('_zscore')])
            
            columns_to_plot = [col for col in numeric_cols if col not in exclude_cols]
            
            # Limit to first 5 columns to avoid overcrowding
            if len(columns_to_plot) > 5:
                logger.info(f"Limiting visualization to first 5 columns out of {len(columns_to_plot)}")
                columns_to_plot = columns_to_plot[:5]
                
            title = "Key Process Parameters"
            y_axis_title = "Value"
        
        else:  # 'all' or any other value
            # Select numeric columns for visualization, excluding certain metadata columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            exclude_cols = ['error_index', 'data_index', 'anomaly_score', '파단직전', 'seconds_to_fault']
            exclude_cols.extend([col for col in numeric_cols if col.endswith('_zscore')])
            
            columns_to_plot = [col for col in numeric_cols if col not in exclude_cols]
            
            # Limit to first 5 columns to avoid overcrowding
            if len(columns_to_plot) > 5:
                logger.info(f"Limiting visualization to first 5 columns out of {len(columns_to_plot)}")
                columns_to_plot = columns_to_plot[:5]
                
            title = "Key Process Parameters"
            y_axis_title = "Value"
        
        if not columns_to_plot:
            logger.warning(f"No data available for {metric_type} parameters")
            
            # Fallback: Just show any numeric columns as a last resort
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            exclude_cols = ['error_index', 'data_index', 'anomaly_score', '파단직전', 'seconds_to_fault']
            exclude_cols.extend([col for col in numeric_cols if col.endswith('_zscore')])
            
            columns_to_plot = [col for col in numeric_cols if col not in exclude_cols]
            
            if columns_to_plot:
                columns_to_plot = columns_to_plot[:5]  # Limit to first 5
                logger.info(f"Using fallback columns for visualization: {columns_to_plot}")
                title = f"Available Numeric Parameters (No {metric_type} found)"
            else:
                return go.Figure().update_layout(title=f"No data available for {metric_type} parameters")
        
        # Create a time window filter if seconds_to_fault exists
        if 'seconds_to_fault' in df.columns:
            min_time, max_time = time_window_mins[0] * 60, time_window_mins[1] * 60
            df_plot = df[(df['seconds_to_fault'] >= min_time) & (df['seconds_to_fault'] <= max_time)]
            x_axis = 'seconds_to_fault'
            x_axis_title = "Time to Fault (seconds)"
        else:
            df_plot = df
            x_axis = config["timestamp_column"]
            x_axis_title = "Timestamp"
        
        if df_plot.empty:
            logger.warning(f"No data in selected time window for {metric_type} parameters")
            return go.Figure().update_layout(title=f"No data in selected time window for {metric_type} parameters")
        
        # Downsample if dataset is too large
        if len(df_plot) > config["max_points_visualization"]:
            original_size = len(df_plot)
            df_plot = downsample_dataframe(
                df_plot, 
                config["max_points_visualization"], 
                config["sampling_method"]
            )
            logger.info(f"Downsampled from {original_size} to {len(df_plot)} points for visualization")
        
        # Create plot with plotly
        fig = go.Figure()
        
        # Add each metric as a line
        for column in columns_to_plot:
            display_name = COLUMN_MAPPING.get(column, column)
            fig.add_trace(go.Scatter(
                x=df_plot[x_axis],
                y=df_plot[column],
                mode='lines',
                name=display_name
            ))
        
        # Add anomaly score if available and requested
        if show_anomaly and 'anomaly_score' in df_plot.columns:
            # Normalize anomaly score to match scale of other parameters for visibility
            max_metric = max([df_plot[col].max() for col in columns_to_plot if not pd.isna(df_plot[col].max())])
            
            if df_plot['anomaly_score'].max() > 0:
                normalized_anomaly = df_plot['anomaly_score'] * (max_metric / df_plot['anomaly_score'].max())
            else:
                normalized_anomaly = df_plot['anomaly_score']
            
            fig.add_trace(go.Scatter(
                x=df_plot[x_axis],
                y=normalized_anomaly,
                mode='lines',
                name='Anomaly Score (Normalized)',
                line=dict(color='red', width=2, dash='dot')
            ))
        
        # Add vertical line at fault time if applicable
        if x_axis == 'seconds_to_fault':
            fig.add_vline(
                x=0, line_width=2, line_dash="dash", line_color="red",
                annotation_text="Fault", annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            legend_title="Parameters",
            template="plotly_white",
            hovermode="x unified"
        )
        
        logger.info(f"Plot created for {metric_type} parameters with {len(df_plot)} data points")
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting key metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return go.Figure().update_layout(title=f"Error plotting metrics: {str(e)}")

def plot_correlation_heatmap(df, parameters=None, config=CONFIG):
    """
    Generate a correlation heatmap for selected parameters with error handling
    """
    try:
        if df is None or df.empty:
            logger.warning("No data available for correlation analysis")
            return go.Figure().update_layout(title="No data available for correlation analysis")
        
        # If no parameters are specified, use available numeric columns
        if parameters is None or not parameters:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Exclude metadata and calculated columns
            exclude_cols = ['error_index', 'data_index', 'anomaly_score', '파단직전', 'seconds_to_fault']
            exclude_cols.extend([col for col in numeric_cols if col.endswith('_zscore')])
            
            parameters = [col for col in numeric_cols if col not in exclude_cols]
            
            # Limit to a manageable number for correlation analysis
            if len(parameters) > 10:
                logger.info(f"Limiting correlation analysis to first 10 numeric columns out of {len(parameters)}")
                parameters = parameters[:10]
        
        # Make sure all columns exist in the dataframe
        parameters = [col for col in parameters if col in df.columns]
        
        if not parameters or len(parameters) < 2:
            logger.warning("Insufficient parameters for correlation analysis")
            return go.Figure().update_layout(title="Insufficient parameters for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = df[parameters].corr()
        
        # Create a heatmap with plotly
        heatmap_data = go.Heatmap(
            z=corr_matrix.values,
            x=[COLUMN_MAPPING.get(p, p) for p in corr_matrix.columns],
            y=[COLUMN_MAPPING.get(p, p) for p in corr_matrix.index],
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='%{y} × %{x}: %{z:.3f}<extra></extra>'
        )
        
        fig = go.Figure(data=heatmap_data)
        
        fig.update_layout(
            title="Parameter Correlation Analysis",
            height=600,
            width=700,
            xaxis={'side': 'bottom', 'tickangle': 45},
            yaxis={'autorange': 'reversed'},
            margin=dict(l=50, r=50, b=100, t=50, pad=4)
        )
        
        logger.info(f"Correlation heatmap created for {len(parameters)} parameters")
        return fig
    
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {str(e)}")
        logger.error(traceback.format_exc())
        return go.Figure().update_layout(title=f"Error generating correlation heatmap: {str(e)}")

def plot_anomaly_timeline(df, threshold_percentile=95, config=CONFIG):
    """
    Generate a timeline plot highlighting anomalies with error handling
    """
    try:
        if df is None or df.empty:
            logger.warning("No data available for anomaly timeline")
            return go.Figure().update_layout(title="No data available for anomaly timeline")
        
        if 'anomaly_score' not in df.columns:
            logger.warning("No anomaly score column found in data")
            return go.Figure().update_layout(title="No anomaly score data available. Run anomaly detection first.")
        
        # Determine the anomaly threshold based on percentile
        threshold = np.percentile(df['anomaly_score'], threshold_percentile)
        logger.info(f"Anomaly threshold at {threshold_percentile}th percentile: {threshold:.4f}")
        
        # Check if we have seconds_to_fault for x-axis
        if 'seconds_to_fault' in df.columns:
            x_axis = 'seconds_to_fault'
            x_axis_title = "Time to Fault (seconds)"
            # Add vertical line at fault time
            add_fault_line = True
        else:
            x_axis = config["timestamp_column"]
            x_axis_title = "Timestamp"
            add_fault_line = False
        
        # Downsample if dataset is too large
        if len(df) > config["max_points_visualization"]:
            original_size = len(df)
            df = downsample_dataframe(
                df, 
                config["max_points_visualization"], 
                config["sampling_method"]
            )
            logger.info(f"Downsampled from {original_size} to {len(df)} points for visualization")
        
        # Create plot
        fig = go.Figure()
        
        # Add anomaly score line
        fig.add_trace(go.Scatter(
            x=df[x_axis],
            y=df['anomaly_score'],
            mode='lines',
            name='Anomaly Score',
            line=dict(color='blue')
        ))
        
        # Add threshold line
        fig.add_trace(go.Scatter(
            x=df[x_axis],
            y=[threshold] * len(df),
            mode='lines',
            name=f'Threshold ({threshold:.2f})',
            line=dict(color='red', dash='dash')
        ))
        
        # Highlight anomalies
        anomalies = df[df['anomaly_score'] > threshold]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies[x_axis],
                y=anomalies['anomaly_score'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
            logger.info(f"Detected {len(anomalies)} anomalies above threshold {threshold:.4f}")
        
        # Add fault line if applicable
        if add_fault_line:
            fig.add_vline(
                x=0, line_width=2, line_dash="dash", line_color="red",
                annotation_text="Fault", annotation_position="top right"
            )
        
        fig.update_layout(
            title="Anomaly Detection Timeline",
            xaxis_title=x_axis_title,
            yaxis_title="Anomaly Score",
            template="plotly_white",
            hovermode="closest"
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting anomaly timeline: {str(e)}")
        logger.error(traceback.format_exc())
        return go.Figure().update_layout(title=f"Error plotting anomaly timeline: {str(e)}")

def plot_parameter_distribution(df, parameter, fault_highlight=True, config=CONFIG):
    """
    Generate a distribution plot for a parameter, highlighting fault vs non-fault periods
    with error handling
    """
    try:
        if df is None or df.empty:
            logger.warning(f"No data available for {parameter} distribution")
            return go.Figure().update_layout(title=f"No data available for {parameter}")
        
        if parameter not in df.columns:
            logger.warning(f"Parameter {parameter} not found in data")
            return go.Figure().update_layout(title=f"Parameter {parameter} not found in data")
        
        # Check if we have fault data
        has_fault_data = config["fault_column"] in df.columns and 1 in df[config["fault_column"]].values
        
        if fault_highlight and has_fault_data:
            # Split data into fault and non-fault periods
            pre_fault_data = df[df[config["fault_column"]] == 0][parameter]
            fault_data = df[df[config["fault_column"]] == 1][parameter]
            
            logger.info(f"Creating distribution plot for {parameter}. Non-fault: {len(pre_fault_data)}, Fault: {len(fault_data)}")
            
            # Create histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=pre_fault_data,
                name='Normal Operation',
                opacity=0.7,
                marker_color='blue',
                nbinsx=30
            ))
            
            fig.add_trace(go.Histogram(
                x=fault_data,
                name='Fault Period',
                opacity=0.7,
                marker_color='red',
                nbinsx=30
            ))
            
            fig.update_layout(
                barmode='overlay',
                title=f"Distribution of {COLUMN_MAPPING.get(parameter, parameter)}",
                xaxis_title=parameter,
                yaxis_title="Count",
                template="plotly_white"
            )
        else:
            # Just show overall distribution
            logger.info(f"Creating overall distribution plot for {parameter} with {len(df[parameter])} data points")
            
            fig = go.Figure(data=go.Histogram(
                x=df[parameter],
                nbinsx=30,
                marker_color='blue'
            ))
            
            fig.update_layout(
                title=f"Distribution of {COLUMN_MAPPING.get(parameter, parameter)}",
                xaxis_title=parameter,
                yaxis_title="Count",
                template="plotly_white"
            )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting parameter distribution: {str(e)}")
        logger.error(traceback.format_exc())
        return go.Figure().update_layout(title=f"Error plotting distribution: {str(e)}")

def create_fault_summary(df, error_index, config=CONFIG):
    """
    Create a summary of fault information with error handling
    """
    try:
        if df is None or df.empty:
            logger.warning("No data available for summary")
            return "No data available for summary"
        
        # Check if we have fault data
        has_fault = 'seconds_to_fault' in df.columns
        
        summary = []
        summary.append(f"### Fault Analysis: Error Index {error_index}")
        summary.append(f"Total Records: {len(df)}")
        
        if has_fault:
            # Find the fault time
            fault_rows = df[df['seconds_to_fault'] == 0]
            if not fault_rows.empty:
                fault_time = fault_rows[config["timestamp_column"]].iloc[0]
                total_duration = abs(df['seconds_to_fault'].min()) + df['seconds_to_fault'].max()
                
                summary.append(f"Fault Time: {fault_time}")
                summary.append(f"Total Duration: {total_duration:.2f} seconds")
                summary.append(f"Time Before Fault: {abs(df['seconds_to_fault'].min()):.2f} seconds")
                summary.append(f"Time After Fault: {df['seconds_to_fault'].max():.2f} seconds")
                
                # Parameter statistics before fault
                pre_fault = df[df['seconds_to_fault'] < 0]
                if not pre_fault.empty:
                    # Get key numerical parameters for statistics
                    numeric_cols = pre_fault.select_dtypes(include=['number']).columns.tolist()
                    exclude_cols = ['error_index', 'data_index', 'anomaly_score', '파단직전', 'seconds_to_fault']
                    exclude_cols.extend([col for col in numeric_cols if col.endswith('_zscore')])
                    key_params = [col for col in numeric_cols if col not in exclude_cols]
                    
                    # Limit to top 5 parameters
                    if len(key_params) > 5:
                        key_params = key_params[:5]
                    
                    if key_params:
                        summary.append("\n### Key Parameter Statistics Before Fault:")
                        for param in key_params:
                            display_name = COLUMN_MAPPING.get(param, param)
                            summary.append(f"{display_name}:")
                            summary.append(f"  - Mean: {pre_fault[param].mean():.2f}")
                            summary.append(f"  - Min: {pre_fault[param].min():.2f}")
                            summary.append(f"  - Max: {pre_fault[param].max():.2f}")
                            summary.append(f"  - Std Dev: {pre_fault[param].std():.2f}")
                    
                    # Add anomaly information if available
                    if 'anomaly_score' in pre_fault.columns:
                        summary.append("\n### Anomaly Analysis:")
                        # Find peak anomaly time before fault
                        peak_anomaly_idx = pre_fault['anomaly_score'].idxmax()
                        peak_anomaly_time = pre_fault.loc[peak_anomaly_idx, 'seconds_to_fault']
                        peak_anomaly_score = pre_fault.loc[peak_anomaly_idx, 'anomaly_score']
                        
                        summary.append(f"Peak Anomaly Score: {peak_anomaly_score:.4f}")
                        summary.append(f"Peak Anomaly Time: {peak_anomaly_time:.2f} seconds before fault")
                        summary.append(f"Mean Anomaly Score: {pre_fault['anomaly_score'].mean():.4f}")
                        
                        # Calculate when anomaly score started rising significantly (e.g., above 75th percentile)
                        anomaly_threshold = np.percentile(pre_fault['anomaly_score'], 75)
                        early_anomalies = pre_fault[pre_fault['anomaly_score'] > anomaly_threshold]
                        if not early_anomalies.empty:
                            earliest_anomaly = early_anomalies['seconds_to_fault'].min()
                            summary.append(f"First Significant Anomaly: {earliest_anomaly:.2f} seconds before fault")
            else:
                summary.append("Warning: No exact fault time point found in data")
        else:
            summary.append("No fault condition detected in this dataset")
        
        logger.info(f"Created fault summary for error index {error_index}")
        return "\n".join(summary)
    
    except Exception as e:
        logger.error(f"Error creating fault summary: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error creating fault summary: {str(e)}"

# Main Gradio interface
def create_interface():
    # Define the interface components
    with gr.Blocks(title="Industrial Process Fault Analysis") as app:
        gr.Markdown("# Industrial Process Fault Analysis Dashboard")
        gr.Markdown("Upload a CSV file with industrial process data to analyze fault patterns.")
        
        # File upload and error index selection
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(label="Upload Data CSV")
                load_button = gr.Button("Load Data", variant="primary")
                data_info = gr.Textbox(label="Data Status", interactive=False)
            
            with gr.Column(scale=1):
                error_index_dropdown = gr.Dropdown(label="Select Error Index", interactive=True, choices=[])
                load_error_button = gr.Button("Analyze Selected Error", variant="primary")
                error_info = gr.Textbox(label="Error Analysis Status", interactive=False)
        
        # Time window selection
        with gr.Row():
            time_window_slider = gr.Slider(
                minimum=-60,
                maximum=60,
                value=[-5, 5],
                step=1,
                label="Time Window (Minutes from Fault)",
                interactive=True
            )
            apply_window_button = gr.Button("Apply Time Window")
        
        # Tabs for different analyses
        with gr.Tabs():
            # Main metrics tab
            with gr.TabItem("Key Metrics"):
                with gr.Row():
                    metric_type = gr.Radio(
                        ["tension", "pressure", "gap", "dancer", "speed", "diff", "all"],
                        label="Metric Type",
                        value="tension"
                    )
                    show_anomaly = gr.Checkbox(label="Show Anomaly Score", value=True)
                
                metrics_plot = gr.Plot(label="Key Metrics")
                update_metrics_button = gr.Button("Update Metrics Plot")
            
            # Correlation analysis tab
            with gr.TabItem("Correlation Analysis"):
                correlation_plot = gr.Plot(label="Parameter Correlations")
                update_correlation_button = gr.Button("Update Correlation Analysis")
            
            # Anomaly detection tab
            with gr.TabItem("Anomaly Detection"):
                with gr.Row():
                    anomaly_method = gr.Radio(
                        ["zscore", "pca"],
                        label="Anomaly Detection Method",
                        value="zscore"
                    )
                    anomaly_threshold = gr.Slider(
                        minimum=80,
                        maximum=99,
                        value=95,
                        step=1,
                        label="Anomaly Threshold Percentile"
                    )
                
                anomaly_plot = gr.Plot(label="Anomaly Timeline")
                update_anomaly_button = gr.Button("Run Anomaly Detection")
            
            # Parameter distribution tab
            with gr.TabItem("Parameter Distribution"):
                with gr.Row():
                    # Initialize with empty choices, will be populated later
                    parameter_dropdown = gr.Dropdown(
                        label="Select Parameter",
                        interactive=True,
                        choices=[]
                    )
                    highlight_fault = gr.Checkbox(label="Highlight Fault Period", value=True)
                
                distribution_plot = gr.Plot(label="Parameter Distribution")
                update_distribution_button = gr.Button("Update Distribution")
            
            # Summary tab
            with gr.TabItem("Fault Summary"):
                summary_text = gr.Markdown(label="Fault Summary")
                update_summary_button = gr.Button("Generate Summary")
        
        # Store the dataframe as state
        dataframe = gr.State(value=None)
        processed_df = gr.State(value=None)
        parameters_list = gr.State(value=[])
        
        # Functions for handling events
        def handle_file_upload(file):
            if file is None:
                return None, [], [], "No file uploaded"
            
            try:
                # Load the data
                df, success, message = load_data(file.name)
                
                if success:
                    # Get unique error indices and ensure they're strings
                    error_indices = sorted(df[CONFIG["error_index_column"]].unique())
                    error_indices = [str(int(idx)) for idx in error_indices if not pd.isna(idx)]
                    
                    # Get parameters for dropdown
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    exclude_cols = ['error_index', 'data_index', 'anomaly_score', '파단직전']
                    param_list = [col for col in numeric_cols if col not in exclude_cols]
                    
                    # Update dropdown choices
                    choices = gr.update(choices=error_indices, value=error_indices[0] if error_indices else None)
                    
                    if not error_indices:
                        return df, choices, param_list, "Data loaded but no valid error indices found."
                    
                    return df, choices, param_list, f"{message}\nFound {len(error_indices)} error indices."
                else:
                    return None, gr.update(choices=[]), [], message
            except Exception as e:
                logger.error(f"Error in file upload handler: {str(e)}")
                return None, gr.update(choices=[]), [], f"Error processing file: {str(e)}"
        
        def handle_error_selection(df, error_index):
            if df is None:
                return None, "No data loaded"
            
            if error_index is None or error_index == "":
                return None, "No error index selected"
            
            try:
                # Convert error_index to integer for processing
                error_index = int(float(error_index))  # Handle both string and float inputs
                
                # Process the selected error index
                processed, has_fault, fault_time, message = process_error_index(df, error_index)
                
                if processed.empty:
                    return None, f"No data found for error index {error_index}"
                    
                if has_fault:
                    # Add anomaly scores
                    processed, _ = calculate_anomaly_scores(processed)
                
                return processed, message
            except ValueError as ve:
                logger.error(f"Invalid error index format: {error_index}")
                return None, f"Invalid error index format: {error_index}"
            except Exception as e:
                logger.error(f"Error in error selection handler: {str(e)}")
                logger.error(traceback.format_exc())
                return None, f"Error processing error index {error_index}: {str(e)}"
        
        def handle_time_window(df, time_window):
            if df is None:
                return df, "No data loaded"
            
            if 'seconds_to_fault' not in df.columns:
                return df, "Cannot apply time window without fault reference point"
            
            try:
                # Convert minutes to seconds
                time_window_seconds = [t * 60 for t in time_window]
                
                # Filter by time window
                filtered_df, message = filter_by_time_window(df, time_window_seconds)
                
                return filtered_df, message
            except Exception as e:
                logger.error(f"Error in time window handler: {str(e)}")
                logger.error(traceback.format_exc())
                return df, f"Error applying time window: {str(e)}"
        
        def update_metrics_visualization(df, metric_type, time_window, show_anomaly):
            if df is None:
                return go.Figure().update_layout(title="No data available")
            
            try:
                return plot_key_metrics(df, metric_type, time_window, show_anomaly)
            except Exception as e:
                logger.error(f"Error in metrics visualization: {str(e)}")
                logger.error(traceback.format_exc())
                return go.Figure().update_layout(title=f"Error creating visualization: {str(e)}")
        
        def update_correlation_visualization(df):
            if df is None:
                return go.Figure().update_layout(title="No data available")
            
            try:
                return plot_correlation_heatmap(df)
            except Exception as e:
                logger.error(f"Error in correlation visualization: {str(e)}")
                logger.error(traceback.format_exc())
                return go.Figure().update_layout(title=f"Error creating correlation heatmap: {str(e)}")
        
        def update_anomaly_visualization(df, method, threshold):
            if df is None:
                return go.Figure().update_layout(title="No data available")
            
            try:
                # Run anomaly detection
                df_with_anomalies, _ = calculate_anomaly_scores(df, method)
                
                return plot_anomaly_timeline(df_with_anomalies, threshold)
            except Exception as e:
                logger.error(f"Error in anomaly visualization: {str(e)}")
                logger.error(traceback.format_exc())
                return go.Figure().update_layout(title=f"Error creating anomaly visualization: {str(e)}")
        
        def update_distribution_visualization(df, parameter, highlight):
            if df is None:
                return go.Figure().update_layout(title="No data available")
            
            if parameter is None or parameter == "":
                return go.Figure().update_layout(title="No parameter selected")
            
            try:
                return plot_parameter_distribution(df, parameter, highlight)
            except Exception as e:
                logger.error(f"Error in distribution visualization: {str(e)}")
                logger.error(traceback.format_exc())
                return go.Figure().update_layout(title=f"Error creating distribution plot: {str(e)}")
        
        def update_summary(df, error_index):
            if df is None:
                return "No data available for summary"
            
            try:
                return create_fault_summary(df, error_index)
            except Exception as e:
                logger.error(f"Error creating summary: {str(e)}")
                logger.error(traceback.format_exc())
                return f"Error creating summary: {str(e)}"
        
        # Connect events
        load_button.click(
            handle_file_upload,
            inputs=[file_input],
            outputs=[dataframe, error_index_dropdown, parameters_list, data_info]
        )
        
        load_error_button.click(
            handle_error_selection,
            inputs=[dataframe, error_index_dropdown],
            outputs=[processed_df, error_info]
        )
        
        apply_window_button.click(
            handle_time_window,
            inputs=[processed_df, time_window_slider],
            outputs=[processed_df, error_info]
        )
        
        update_metrics_button.click(
            update_metrics_visualization,
            inputs=[processed_df, metric_type, time_window_slider, show_anomaly],
            outputs=[metrics_plot]
        )
        
        update_correlation_button.click(
            update_correlation_visualization,
            inputs=[processed_df],
            outputs=[correlation_plot]
        )
        
        update_anomaly_button.click(
            update_anomaly_visualization,
            inputs=[processed_df, anomaly_method, anomaly_threshold],
            outputs=[anomaly_plot]
        )
        
        # Update parameter dropdown choices when parameter list changes
        parameters_list.change(
            lambda params: gr.update(choices=params if params else []),  # Add fallback empty list
            inputs=[parameters_list],
            outputs=[parameter_dropdown]
        )
        
        update_distribution_button.click(
            update_distribution_visualization,
            inputs=[processed_df, parameter_dropdown, highlight_fault],
            outputs=[distribution_plot]
        )
        
        update_summary_button.click(
            update_summary,
            inputs=[processed_df, error_index_dropdown],
            outputs=[summary_text]
        )
        
        # Define automatic updates to improve user experience
        error_index_dropdown.change(
            handle_error_selection,
            inputs=[dataframe, error_index_dropdown],
            outputs=[processed_df, error_info]
        )
        
        # Run update metrics automatically when processed_df changes
        processed_df.change(
            update_metrics_visualization,
            inputs=[processed_df, metric_type, time_window_slider, show_anomaly],
            outputs=[metrics_plot]
        )
    
    return app

# Entry point for the application
if __name__ == "__main__":
    # Configure memory optimization based on available resources
    try:
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory < 2:
            logger.warning(f"Low available memory detected: {available_memory:.2f} GB")
            # Adjust configuration for low-memory environments
            CONFIG["chunk_size"] = 50000
            CONFIG["max_points_visualization"] = 5000
        elif available_memory > 8:
            logger.info(f"Abundant memory detected: {available_memory:.2f} GB")
            # Increase chunk size for faster processing on high-memory systems
            CONFIG["chunk_size"] = 200000
            CONFIG["max_points_visualization"] = 20000
    except Exception as e:
        logger.warning(f"Error checking memory resources: {str(e)}. Using default settings.")
    
    logger.info(f"Starting application with configuration: {CONFIG}")
    app = create_interface()
    app.launch(share=False)  # Set share=False for production deployment