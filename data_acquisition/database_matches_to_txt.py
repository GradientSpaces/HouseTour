import sqlite3

# Path to the COLMAP database file
database_path = 'database_vocab.db'

# Connect to the database
conn = sqlite3.connect(database_path)
cursor = conn.cursor()


cursor.execute("SELECT * FROM images")
image_ids = cursor.fetchall()
image_ids = {image_id[0]: image_id[1] for image_id in image_ids}

# Query to get all the matches
cursor.execute("SELECT * FROM two_view_geometries")

# Open a file to write the matches
with open('vocab_matches.txt', 'w') as file:
    for row in cursor.fetchall():
        pair_id = row[0]
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) // 2147483647
        image_id1, image_id2 = image_ids[image_id1], image_ids[image_id2]
        matches = row[1]
        
        # Write the image pair and their matches
        if matches > 15:
            file.write(f"{image_id1} {image_id2} {matches}\n")  # Number of matches

# Close the database connection
conn.close()
