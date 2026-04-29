import sqlite3
import readline
import config
def get_unique_names(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT name FROM faces")
    return [row[0] for row in cursor.fetchall()]


def count_embeddings(conn, name):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM faces WHERE name = ?", (name,))
    return cursor.fetchone()[0]


def rename_person(conn, old_name, new_name):
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE faces SET name = ? WHERE name = ?",
        (new_name, old_name)
    )
    conn.commit()
    return cursor.rowcount


def safe_input(prompt):
    value = input(prompt).strip()
    if value.lower() == 'q':
        print("Exiting safely.")
        return None
    return value

def setup_autocomplete(options):
    def completer(text, state):
        buffer = readline.get_line_buffer()
        matches = [name for name in options if name.lower().startswith(buffer.lower())]
        if state < len(matches):
            return matches[state]
        return None

    readline.set_completer(completer)
    readline.parse_and_bind("set editing-mode emacs")
    readline.parse_and_bind("tab: complete")

# Connect to the database
conn = sqlite3.connect(config.DB_PATH)

print("\nCurrent names in database:")
names = get_unique_names(conn)
print(names)

while True:
    choice = safe_input("\nDo you want to edit a name? (y/n, q to quit): ")
    if choice is None or choice.lower() != 'y':
        break

    names = get_unique_names(conn)
    setup_autocomplete(names)

    name_to_edit = safe_input("Enter name to edit (TAB for suggestions, q to quit): ")
    if name_to_edit is None:
        break

    if not name_to_edit:
        print("Name cannot be empty.")
        continue

    if name_to_edit not in names:
        print(f"Name '{name_to_edit}' not found in database.")
        continue

    count = count_embeddings(conn, name_to_edit)
    print(f"'{name_to_edit}' has {count} embeddings.")

    new_name = safe_input("Enter the new name (or q to quit): ")
    if new_name is None:
        break

    if not new_name:
        print("New name cannot be empty.")
        continue

    confirm = safe_input(
        f"Are you sure you want to rename '{name_to_edit}' to '{new_name}'? (y/n): "
    )
    if confirm is None or confirm.lower() != 'y':
        print("Operation cancelled.")
        continue

    updated_rows = rename_person(conn, name_to_edit, new_name)
    print(f"Successfully updated {updated_rows} rows.")

    # Refresh names list
    names = get_unique_names(conn)

print("\nFinal names in database:")
print(get_unique_names(conn))

conn.close()
print("Database connection closed.")