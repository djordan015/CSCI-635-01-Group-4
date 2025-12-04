import pandas as pd

# Load dataset
data = pd.read_csv("../data/processed/spotify_features.csv")

# real genre → macro-genre mapping dictionary
genre_mapping = {
    # Acoustic/Calm
    'ambient': 'Acoustic',
    'acoustic': 'Acoustic',
    'chill': 'Acoustic',
    'sleep': 'Acoustic',
    'new-age': 'Acoustic',
    'classical': 'Acoustic',
    'piano': 'Acoustic',
    'guitar': 'Acoustic',
    'opera': 'Acoustic',

    # Electronic/Dance
    'deep-house': 'Electronic',
    'minimal-techno': 'Electronic',
    'drum-and-bass': 'Electronic',
    'club': 'Electronic',
    'dance': 'Electronic',
    'disco': 'Electronic',
    'techno': 'Electronic',
    'house': 'Electronic',
    'edm': 'Electronic',
    'trance': 'Electronic',
    'dubstep': 'Electronic',
    'progressive-house': 'Electronic',
    'chicago-house': 'Electronic',
    'detroit-techno': 'Electronic',
    'electro': 'Electronic',
    'hardstyle': 'Electronic',
    'breakbeat': 'Electronic',
    'garage': 'Electronic',
    'trip-hop': 'Electronic',
    'electronic': 'Electronic',
    'dub': 'Electronic',

    # Hip-Hop/Funk
    'hip-hop': 'Hip-Hop',
    'funk': 'Hip-Hop',
    'groove': 'Hip-Hop',
    'soul': 'Hip-Hop',
    'afrobeat': 'Hip-Hop',
    'dancehall': 'Hip-Hop',

    # Rock
    'rock': 'Rock',
    'alt-rock': 'Rock',
    'hard-rock': 'Rock',
    'rock-n-roll': 'Rock',
    'punk': 'Rock',
    'punk-rock': 'Rock',
    'psych-rock': 'Rock',
    'power-pop': 'Rock',
    'ska': 'Rock',

    # Metal
    'metal': 'Metal',
    'black-metal': 'Metal',
    'death-metal': 'Metal',
    'heavy-metal': 'Metal',
    'grindcore': 'Metal',
    'hardcore': 'Metal',
    'industrial': 'Metal',
    'goth': 'Metal',
    'emo': 'Metal',
    'metalcore': 'Metal',

    # Latin
    'salsa': 'Latin',
    'samba': 'Latin',
    'sertanejo': 'Latin',
    'forro': 'Latin',
    'tango': 'Latin',
    'spanish': 'Latin',

    # Folk/Country
    'folk': 'Folk',
    'country': 'Folk',
    'blues': 'Folk',

    # Jazz
    'jazz': 'Jazz',
    'gospel': 'Jazz',

    # Pop/Other
    'pop': 'Pop',
    'pop-film': 'Pop',
    'party': 'Pop',
    'romance': 'Pop',
    'sad': 'Pop',
    'show-tunes': 'Pop',
    'comedy': 'Pop',
    'k-pop': 'Pop',
    'indie-pop': 'Pop',
    'cantopop': 'Pop',
}

# Map genres to macro-genres → new column
data['macro_genre'] = data['genre'].map(genre_mapping)

# Keep only mapped rows
final_dataset = data.dropna(subset=['macro_genre'])

# Export CSV
final_dataset.to_csv("../data/processed/final_dataset.csv", index=False)

# Print stats
print("Total mapped samples:", len(final_dataset))
print("Macro-genre distribution:")
print(final_dataset['macro_genre'].value_counts())
