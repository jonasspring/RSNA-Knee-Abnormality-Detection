import os


def prepare_study_dict(train_series_df, series_dir):
    """
    Generates a dict containing per study one series for each imaging plane
    """
    
    def count_dcm_files_per_folder(subfolders):
        dcm_count_dict = {}
        for subfolder_path in subfolders:
            if os.path.isdir(subfolder_path):
                dcm_count_dict[subfolder_path.name] = len(list(subfolder_path.glob('*.dcm')))
                
        return dcm_count_dict

    study_dict = {}
    
    # group by id and plane
    for study_id, study_group in train_series_df.groupby('StudyInstanceUID'):
        study_dict[study_id] = {}
        for plane in ['Sagittal', 'Coronal', 'Axial']:
            plane_df = study_group[study_group['Anatomical_Plane'] == plane]
            if not plane_df.empty:
                # Prioritize fluid sentitive MRIs
                if (plane_df['Fluid_Sensitive'] == 1).any():
                    plane_df = plane_df.loc[plane_df['Fluid_Sensitive'] == 1]

                # Use MRI series with most dcm files
                if len(plane_df.index) > 1:
                    subfolders = [series_dir / study_id / series_id for series_id in plane_df['SeriesInstanceUID'].values]
                    dcm_count_dict = count_dcm_files_per_folder(subfolders)
                    best_series, _ = max(dcm_count_dict.items(), key=lambda item: item[1])
                else:
                    best_series = plane_df.iloc[0]['SeriesInstanceUID'] 
                    
                study_dict[study_id][plane] = best_series
            else:
                study_dict[study_id][plane] = None # Fallback
                
    return study_dict
