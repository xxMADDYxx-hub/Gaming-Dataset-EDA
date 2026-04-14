import pandas as pd
import plotly.express as px

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(path):
    try:
        df = pd.read_csv(path)
        print("Dataset loaded successfully\n")
        return df
    except Exception as e:
        print("Error loading dataset:", e)
        return None


# -------------------------------
# EXPLORE DATA
# -------------------------------
def explore_data(df):
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nSample:\n", df.head())


# -------------------------------
# CLEAN DATA
# -------------------------------
def clean_data(df):
    print("\nCleaning data...")

    df = df.dropna()

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    return df


# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
def feature_engineering(df):
    print("\nCreating features...")

    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['year'] = df['release_date'].dt.year

    return df


# -------------------------------
# ANALYSIS FUNCTIONS
# -------------------------------
def get_top_genres(df):
    return df['genre'].value_counts().head(5).reset_index()


def get_games_per_year(df):
    return df['year'].value_counts().sort_index().reset_index()


def get_top_publishers(df):
    return df['publisher'].value_counts().head(5).reset_index()


def get_price_stats(df):
    return df['price'].describe()


# -------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------

def plot_top_genres(df):
    data = get_top_genres(df)
    data.columns = ['Genre', 'Count']

    fig = px.bar(data, x='Genre', y='Count', color='Genre', text='Count',
                 title="Top 5 Game Genres")
    fig.update_traces(width=0.4)
    fig.show()


def plot_games_per_year(df):
    data = get_games_per_year(df)
    data.columns = ['Year', 'Count']

    fig = px.line(data, x='Year', y='Count', markers=True,
                  title="Games Released Over Time")
    fig.show()


def plot_price_distribution(df):
    fig = px.histogram(df, x='price', nbins=30,
                       title="Game Price Distribution")
    fig.show()

def plot_rating_vs_price(df):
    if 'rating' in df.columns:

        # 🔹 Reduce points (important)
        sample = df.sample(min(1000, len(df)))

        fig = px.scatter(
            sample,
            x='price',
            y='rating',
            color='genre',
            title="Price vs Rating (Sampled for Clarity)"
        )

        # 🔹 Make points smaller + transparent
        fig.update_traces(marker=dict(size=6, opacity=0.5))

        fig.show()
#def plot_rating_vs_price(df):
#    if 'rating' in df.columns:
#       fig = px.scatter(df, x='price', y='rating', color='genre',
#                         title="Price vs Rating")
#        fig.show()#


def plot_top_publishers(df):
    if 'publisher' in df.columns:
        data = get_top_publishers(df)
        data.columns = ['Publisher', 'Count']

        fig = px.bar(data, x='Publisher', y='Count', color='Publisher',
                     text='Count', title="Top Publishers")
        fig.update_traces(width=0.4)
        fig.show()


# -------------------------------
# INSIGHTS
# -------------------------------
def generate_insights(df):
    print("\n--- INSIGHTS ---")

    print("\nTop Genres:")
    print(df['genre'].value_counts().head(5))

    print("\nPrice Statistics:")
    print(df['price'].describe())

    if 'rating' in df.columns:
        print("\nAverage Rating:", df['rating'].mean())

    if 'year' in df.columns:
        print("Peak Release Year:", df['year'].value_counts().idxmax())


# -------------------------------
# MAIN
# -------------------------------
def main():
    df = load_data("gaming_data.csv")
    if df is None:
        return

    explore_data(df)

    df = clean_data(df)
    df = feature_engineering(df)

    plot_top_genres(df)
    plot_games_per_year(df)
    plot_price_distribution(df)
    plot_rating_vs_price(df)
    plot_top_publishers(df)

    generate_insights(df)

    print("\nEDA Completed Successfully")


if __name__ == "__main__":
    main()