import pandas as pd
import os


def add_resultat_to_images(csv_resultats_path, csv_images_path, output_path=None):
    # Lecture des CSV
    df_resultats = pd.read_csv(csv_resultats_path)
    df_images = pd.read_csv(csv_images_path)

    # Merge (on garde toutes les images)
    df_merged = df_images.merge(
        df_resultats[["patient_id", "status"]],
        on="patient_id",
        how="left"
    )

    df_merged = df_merged.dropna(subset=["status"])
    df_merged["status"] = df_merged["status"].astype(int)


    # Sauvegarde si demandé
    if output_path:
        df_merged.to_csv(output_path, index=False)

    return df_merged

clinical_data = "pipe_trident_torchmil\\clinical_data.csv"
annotation = "pipe_trident_torchmil\\annotations.csv"

add_resultat_to_images(clinical_data, annotation, "pipe_trident_torchmil\\dataset_list.csv")

def split_csv(input_path, output, marker_list, image_folder, random_seed=None):
    #input_path = dataframe
    df = input_path

    for marker in marker_list:
        # Filtrer les lignes pour le marqueur actuel
        df_marker = df[df['marker'] == marker]

        #Supprimer ligne fantome pour eviter soucis de lecture de fichier svs
        rows_to_drop = []
        for index, row in df_marker.iterrows():
            filename = row["patient+AF8-id"] + ".svs"
            full_path = os.path.join(image_folder, filename)

            if not os.path.exists(full_path):
                rows_to_drop.append(index)

            df_marker = df_marker.drop(rows_to_drop).reset_index(drop=True)
            
        df_shuffled = df_marker.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Calcul de la séparation
        split_index = int(len(df_shuffled) * 0.8)

        # Division
        train_csv = df_shuffled.iloc[:split_index]
        test_csv = df_shuffled.iloc[split_index:]

        # Sauvegarde CSV
        output_path_80 = os.path.join(output, f"{marker}_train.csv")
        output_path_20 = os.path.join(output, f"{marker}_test.csv")
        
        train_csv.to_csv(output_path_80, index=False)
        test_csv.to_csv(output_path_20, index=False)

        print(f"csv test {marker} créé : {output_path_80}")
        print(f"csv train {marker} créé : {output_path_20}")
        return train_csv, test_csv