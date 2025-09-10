import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px # Import plotly.express for easy plotting
import pandas as pd
import numpy as np

# ---------- Load Data ----------
@st.cache_resource
def load_data():
    df = pd.read_csv("audible_data_cleaned (1).csv")
    hybrid_sim = np.load("hybrid_sim.npy")
    return df, hybrid_sim

df, hybrid_sim = load_data()


# ---------- Recommendation Functions ----------
def recommend_books_by_name(book_name, top_n=5):
    # Case-insensitive partial match
    matches = df[df['Book Name'].str.contains(book_name, case=False, na=False)]

    if matches.empty:
        print(f"'{book_name}' not found in dataset.")
        return None

    # Take the first matching book
    idx = matches.index[0]
    selected_book = df.loc[idx, 'Book Name']

    # Get similarity scores for this book
    sim_scores = list(enumerate(hybrid_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 1]  # skip the book itself

    # Get indices of recommended books
    book_indices = [i[0] for i in sim_scores]

    print(f"Showing recommendations for: {selected_book}\n")
    return df.iloc[book_indices][['Book Name', 'Author', 'Genre', 'Rating', 'Number of Reviews']]

def recommend_books_by_genre(genre, top_n=5):
    genre_books = df[df['Genre'].str.contains(genre, case=False, na=False)]
    if genre_books.empty:
        return pd.DataFrame(columns=["Book Name", "Author", "Genre", "Rating", "Number of Reviews"])
    
    recs = genre_books.sort_values(by=["Rating", "Number of Reviews"], ascending=False).head(top_n)
    recs = recs[['Book Name', 'Author', 'Genre', 'Rating', 'Number of Reviews']]
    return recs.reset_index(drop=True)


# ---------- Streamlit UI ----------

st.header("📚 Intelligent Book Recommendation System")
st.set_page_config(layout="wide")
col1, col2 = st.columns([2, 4], gap = 'large')
with col1:
    page = option_menu(
    menu_title="Main Menu",  # A title is required
    options=["Home", "Data Analysis", "Recommender", "About model performances"],
    icons=["house", "bar-chart", "compass", "book"],  # Optional icons from Bootstrap
    menu_icon="cast",
    default_index=0,
    orientation="vertical",  # Make the menu horizontal
    )
    
    if page == 'Data Analysis':
        
        # Create scatter plot with Plotly
            fig = px.scatter(
                    df,
                    x="Number of Reviews",
                    y="Rating",
                    opacity=0.5,
                    title="Rating vs. Number of Reviews",
                    labels={"Number of Reviews": "Number of Reviews", "Rating": "Rating"}
                )

            # Add gridlines
            fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
            # Show the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
    
with col2:
    if page == 'Home':
        st.write("Designed a book recommendation system that retrieves book details from given datasets, processes and cleans the data before applying NLP techniques and clustering methods and builds multiple recommendation models. The final application will allow users to search for book recommendations using a user-friendly interface deployed with Streamlit and hosted on AWS.")
        st.markdown("""
                    **Business Use Cases:**
                    *   ***Personalized Reading Experience:***
                            Help readers discover books tailored to their preferences based on their reading history, genres, or authors.
                    *   ***Enhanced Library Systems:***
                            Libraries and bookstores can use recommendations to improve book borrowing/sales based on popular or similar reads.
                    *   ***Improved Author/Publisher Targeting:***
                            Provide authors and publishers with data-driven insights about popular genres, reader preferences, and high-demand books.
                    """)
    
    if page == 'Data Analysis':

        da_page = option_menu(
            menu_title="Main Menu",  
            options=["DA - 1", "DA - 2", "Scenario Based"],
            icons=["bar-chart", "line-chart", "pie-chart"],  
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",  
        )

        if da_page == "DA - 1":
            col1, col2 = st.columns([2, 2], gap='large')
            with col1:
                # Count genres
                genre_counts = df['Genre'].value_counts()
                filtered_genre_counts = genre_counts[genre_counts.index != 'Unknown']
                top_genres = filtered_genre_counts.head(10)

                fig = px.bar(
                    top_genres,
                    x=top_genres.index,
                    y=top_genres.values,
                    labels={"x": "Genre", "y": "Number of Books"},
                    title="Top 10 Most Popular Genres (Excluding Unknown)"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                highest_rated_books_by_author = (
                    df.groupby('Author')['Rating']
                    .max()
                    .sort_values(ascending=False)
                    .head(10)
                )
                fig = px.bar(
                    highest_rated_books_by_author,
                    x=highest_rated_books_by_author.index,
                    y=highest_rated_books_by_author.values,
                    labels={"x": "Author", "y": "Highest Rating"},
                    title="Top 10 Highest-Rated Books by Author"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns([2, 2], gap='large')
            with col3:
                fig = px.histogram(
                    df,
                    x="Rating",
                    nbins=20,
                    title="Distribution of Book Ratings",
                    labels={"Rating": "Rating", "count": "Number of Books"}
                )
                fig.update_traces(marker_line_color="black", marker_line_width=1)
                fig.update_layout(yaxis_title="Number of Books", xaxis_title="Rating")
                st.plotly_chart(fig, use_container_width=True)

        elif da_page == "DA - 2":  
            col1, col2 = st.columns([2, 2], gap='large')
            with col1:
                book_pairs = [
                    ("The Fault in Our Stars", "The Diviners", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "The Riddle of the Third Mile: Inspector Morse Mysteries, Book 6", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "The Sense of Style: The Thinking Person’s Guide to Writing in the 21st Century", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "Capital and Ideology", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "Marvel Super Heroes: Secret Wars", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "Six Easy Pieces: Essentials of Physics Explained by Its Most Brilliant Teacher", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "The Lovely Bones", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "Radical Forgiveness: A Revolutionary Five-Stage Process to Heal Relationships, Let Go of Anger and Blame, Find Peace in Any Situation", 1.0000),
                    ("The Meltdown: Diary of a Wimpy Kid, Book 13", "The Book of Hygge: The Danish Art of Living Well", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "Superforecasting: The Art and Science of Prediction", 1.0000),
                    ("The Meltdown: Diary of a Wimpy Kid, Book 13", "Pregnancy (Hindi Edition)", 1.0000),
                    ("Notes for Healthy Kids", "Darwin's Dangerous Idea: Evolution and the Meanings of Life", 1.0000),
                    ("Notes for Healthy Kids", "Crushing It in Apartments and Commercial Real Estate: How a Small Investor Can Make It Big", 1.0000),
                    ("The Bear and the Dragon", "The Mister", 1.0000),
                    ("The Way of the SEAL: Think like an Elite Warrior to Lead and Succeed: Updated and Expanded Edition", 
                     "The Neapolitan Novels: My Brilliant Friend, The Story of a New Name, Those Who Leave and Those Who Stay & The Story of the Lost Child: The Complete BBC Radio Collection", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "Kinds of Minds: Toward an Understanding of Consciousness", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "Hard Luck: Diary of a Wimpy Kid, Book 8", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "The Effortless Experience: Conquering the New Battleground for Customer Loyalty", 1.0000),
                    ("The Meltdown: Diary of a Wimpy Kid, Book 13", "Feel Better Fast and Make It Last: Unlock Your Brain's Healing Potential to Overcome Negativity, Anxiety, Anger, Stress, and Trauma", 1.0000),
                    ("Everyday Arabic for Beginners - 400 Actions & Activities", "We Are Never Meeting in Real Life", 1.0000)
                ]
                df_pairs = pd.DataFrame(book_pairs, columns=["Book 1", "Book 2", "Similarity"])
                st.subheader("📖 Top 20 Most Frequently Clustered Book Pairs")
                st.dataframe(df_pairs, use_container_width=True)
                
            with col2:
                
                st.subheader("📚 How does genre similarity affect book recommendations?")

                # Hybrid Recommendations (TF-IDF + BERT)
                hybrid_data = [
                    [514, "Die Trying: Jack Reacher, Book 2", "Lee Child", "Unknown", 4.4],
                    [1984, "Transcendent Kingdom", "Yaa Gyasi", "Southern United States Literature", 4.3],
                    [2193, "Charlie and the Chocolate Factory", "Roald Dahl", "Fiction Classics for Children", 4.6],
                    [2237, "Beach Read", "Emily Henry", "Romantic Comedy", 4.4],
                    [2278, "Know My Name", "Chanel Miller", "Sexual Abuse & Harassment", 4.8],
                ]
                df_hybrid = pd.DataFrame(hybrid_data, columns=["ID", "Book Name", "Author", "Genre", "Rating"])

                # Enhanced Hybrid Recommendations (TF-IDF + BERT + Genre)
                enhanced_data = [
                    [514, "Die Trying: Jack Reacher, Book 2", "Lee Child", "Unknown", 4.4],
                    [1984, "Transcendent Kingdom", "Yaa Gyasi", "Southern United States Literature", 4.3],
                    [2193, "Charlie and the Chocolate Factory", "Roald Dahl", "Fiction Classics for Children", 4.6],
                    [2237, "Beach Read", "Emily Henry", "Romantic Comedy", 4.4],
                    [2278, "Know My Name", "Chanel Miller", "Sexual Abuse & Harassment", 4.8],
                ]
                df_enhanced = pd.DataFrame(enhanced_data, columns=["ID", "Book Name", "Author", "Genre", "Rating"])
                st.dataframe(df_enhanced, use_container_width=True)
                
                # Calculate average rating and average number of reviews per author
                author_stats = df.groupby('Author').agg(
                    average_rating=('Rating', 'mean'),
                    average_reviews=('Number of Reviews', 'mean')
                ).reset_index()

                st.subheader("📊 Author Statistics vs Rating and Reviews")

                # Scatter plot with Plotly
                fig = px.scatter(
                    author_stats,
                    x="average_reviews",
                    y="average_rating",
                    opacity=0.5,
                    title="Average Rating vs. Average Number of Reviews by Author",
                    labels={"average_reviews": "Average Number of Reviews", "average_rating": "Average Rating"}
                )

                # Add gridlines
                fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)

                # Show the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)

        elif da_page == "Scenario Based":
            col1, col2 = st.columns([2, 2], gap = 'large')
            with col1:
                # Top 5 Recommendations for Thriller Book Lovers
                thriller_recs = [
                    (2425, "Blasphemy: The Trial of Danesh Masih", "Osman Haneef", 4.9),
                    (3125, "Nightshade", "Anthony Horowitz", 4.8),
                    (3002, "Gates of Fire: An Epic Novel of the Battle of ...", "Steven Pressfield", 4.7),
                    (2692, "Murder on the Orient Express (Dramatised)", "Agatha Christie", 4.7),
                    (3162, "Trunk Music: Harry Bosch Series, Book 5", "Michael Connelly", 4.7),
                ]

                # Convert to DataFrame
                df_thriller = pd.DataFrame(thriller_recs, columns=["Book ID", "Book Name", "Author", "Rating"])

                # Show in Streamlit
                st.subheader("🔍 Top 5 Recommendations for Thriller Book Lovers")
                st.dataframe(df_thriller, use_container_width=True) 
                
            with col2:
                # Top 5 Science Fiction Books
                sci_fi_recs = [
                    (2909, "Doctor Who: 10th Doctor Tales: 10th Doctor Aud...", "Peter Anghelides", 4.8),
                    (2786, "Morning Star: Book III of the Red Rising Trilogy", "Pierce Brown", 4.8),
                    (2942, "So Long and Thanks for All the Fish", "Douglas Adams", 4.7),
                    (2117, "Gemina: The Illuminae Files, Book 2", "Amie Kaufman", 4.7),
                    (2266, "The Martian", "Andy Weir", 4.7),
                ]

                # Convert to DataFrame
                df_sci_fi = pd.DataFrame(sci_fi_recs, columns=["Book ID", "Book Name", "Author", "Rating"])

                # Show in Streamlit
                st.subheader("🚀 Top 5 Science Fiction Books")
                st.dataframe(df_sci_fi, use_container_width=True)
                
            col3, = st.columns(1)  # note the comma to unpack single element
            with col3:
                st.subheader("💎 Top 20 Highly-Rated Books with Low Popularity (Hidden Gems)")
            
                # Top 20 Highly-Rated Books with Low Popularity (Hidden Gems)
                hidden_gems = [
                    (19, "Goodness Gracious Me: The Complete Radio Series 1-3", "Meera Syal", 5.0, 1),
                    (22, "Jonathan Van Ness: Audible Sessions: FREE Exclusive Interview", "Holly Newson", 5.0, 1),
                    (66, "Eighty Days to Elsewhere", "KC Dyer", 5.0, 1),
                    (447, "Midnight's Children: BBC Radio 4 full-cast dramatisation", "Salman Rushdie", 5.0, 1),
                    (581, "Gather ‘Round the Sound: Holiday Stories from Beloved Authors and Great Performers Across the Globe", "Paulo Coelho", 5.0, 1),
                    (675, "The Lottery Ticket", "Anton Chekhov", 5.0, 1),
                    (687, "Aamool Kranti Ki Chunauti [Radical Revolution Is the Key Challenge]", "J. Krishnamurti", 5.0, 1),
                    (719, "The Himalayan Arc: Journeys East of South-east", "Namita Gokhale", 5.0, 1),
                    (767, "Free Excerpt: Star Wars: Heir to the Empire - Behind the Scenes", "Timothy Zahn", 5.0, 1),
                    (914, "Introduction to Intermediate French Conversation Lessons", "Audible Inc.", 5.0, 1),
                    (937, "The Spirit of Mantra with Deva Premal & Miten: 21 Chant Practices for Daily Life", "Deva Premal", 5.0, 1),
                    (1018, "Vietnam War", "Maurice Isserman", 5.0, 1),
                    (1031, "Birth of Ganesha", "Shobha Viswanath", 5.0, 1),
                    (1100, "Roll of the Dice: Duryodhana's Mahabharata", "Anand Neelakantan", 5.0, 1),
                    (1260, "Is Jesus Truly God?: How the Bible Teaches the Divinity of Christ", "Gregory R. Lanier", 5.0, 1),
                    (1594, "The Feynman Lectures on Physics: Volume 2, Advanced Quantum Mechanics", "Richard P. Feynman", 5.0, 1),
                    (1650, "Geronimo Stilton #20 and #21: Surf's Up, Geronimo & The Wild Wild West", "Geronimo Stilton", 5.0, 1),
                    (1764, "Learn Thai with Innovative Language's Proven Language System - Level 1: Introduction to Thai: Introduction Thai #2", "Innovative Language Learning", 5.0, 1),
                    (2198, "Manage Your Time, Master Your Life", "Robin Sharma", 5.0, 1),
                    (2273, "No Calculator? No Problem!: Mastering Mental Math", "Art Benjamin", 5.0, 1),
                ]
            
                # Convert to DataFrame
                df_hidden_gems = pd.DataFrame(hidden_gems, columns=["Index", "Book Name", "Author", "Rating", "Number of Reviews"])
            
                # Show in Streamlit
                st.dataframe(df_hidden_gems, use_container_width=True)


                
    if page == 'Recommender':
        st.header("📚 Book Recommender System")
        st.write("Find books based on title or genre.")

        option = st.radio("Choose input type:", ["Book Title", "Genre"], horizontal=True)

        if option == "Book Title":
            col1, col2 = st.columns([2, 3], gap="large")

            with col1:
                with st.form("title_form", clear_on_submit=False):
                    book_name = st.text_input("Enter a book title:")
                    recommend_clicked = st.form_submit_button("Get Recommendations")

            with col2:
                if recommend_clicked:
                    if book_name.strip():
                        results = recommend_books_by_name(book_name, top_n=5)
                        if results.empty:
                            st.warning("⚠️ Book not found in the dataset. Try another title.")
                        else:
                            with st.expander("📖 Recommended Books", expanded=True):
                                st.dataframe(results, use_container_width=True)
                else:
                    st.info("👉 Please enter a book title to get recommendations.")

        elif option == "Genre":
            col1, col2 = st.columns([2, 3], gap="large")

            with col1:
                with st.form("genre_form", clear_on_submit=False):
                    genre = st.text_input("Enter a genre (e.g., Romance, Fantasy, Thriller):")
                    recommend_clicked = st.form_submit_button("Get Recommendations")

            with col2:
                if recommend_clicked:
                    if genre.strip():
                        results = recommend_books_by_genre(genre, top_n=5)
                        if results.empty:
                            st.warning("⚠️ No books found for this genre. Try another one.")
                        else:
                            with st.expander("📖 Recommended Books", expanded=True):
                                st.dataframe(results, use_container_width=True)
                else:
                    st.info("👉 Please enter a genre to get recommendations.")

    if page == 'About model performances':
        
        st.header("Individual Model Results")
        # --- 1. Define the data ---
        tfidf_results = {
        'Precision': 0.814, 'Recall': 0.8708333333333332, 'NDCG': 0.9216761683504505, 'MAP': 0.7836548892208155, 'RMSE': 14.644659903414711,
        'Precision': 0.778, 'Recall': 0.8903809523809524, 'NDCG': 0.8999207538338473, 'MAP': 0.7629160185345921, 'RMSE': 14.478680215659242
        }

        bert_results = {
        'Precision': 0.7619999999999999, 'Recall': 0.7923333333333332, 'NDCG': 0.8660197505073516, 'MAP': 0.7362393768610817, 'RMSE': 13.325822521021452,
        'Precision': 0.6999999999999998, 'Recall': 0.7456666666666666, 'NDCG': 0.8355214056623698, 'MAP': 0.7055344190744853, 'RMSE': 13.479724292024255
        }

        hybrid_results = {
        'Precision': 0.762, 'Recall': 0.7908333333333332, 'NDCG': 0.8744867183866972, 'MAP': 0.7435302683540624, 'RMSE': 17.10638005521974,
        'Precision': 0.741, 'Recall': 0.8046904761904763, 'NDCG': 0.8845009694450383, 'MAP': 0.7412433081858845, 'RMSE': 16.83675496692056
        }

        # --- 2. Create the DataFrame ---
        all_results = {
            'TF-IDF': tfidf_results,
            'BERT': bert_results,
            'Hybrid': hybrid_results
        }

        df = pd.DataFrame.from_dict(all_results, orient='index')
        df.index.name = 'Model'

        
        # Select a metric for comparison
        metric_choice = st.selectbox(
        "Select a metric to compare:",
        options=df.columns.to_list(),
        index=df.columns.get_loc('Precision') # Optional: Set a default metric
        )
        # --- 4. Create the Plotly bar chart ---
        # Reset the index to make "Model" a regular column for Plotly
        plot_df = df.reset_index()
        fig = px.bar(
            plot_df,
            x='Model',
            y=metric_choice,
            title=f"Comparison of {metric_choice}",
            labels={'Model': 'Model', metric_choice: 'Score'},
            color='Model',
            text_auto='.3f' # Display values on the bars, formatted to 3 decimal places
        )
        # Optional: Add formatting to the chart
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title=metric_choice,
            title_x=0.5, # Center the title
            font=dict(size=14)
        )
        # Display the interactive Plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)