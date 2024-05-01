from flask import Flask, render_template, request
import pandas as pd
import webbrowser

app = Flask(__name__)

data = pd.read_csv('ConvertedData.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods=['POST'])
def search():
    # Get the search query from the form
    query = request.form['query']
 
    # convert topics column to string type
    data['topics'] = data['topics'].astype(str)

    data['keyphrases'] = data['keyphrases'].astype(str)
    
    # Filter the DataFrame based on the query in both keyphrases and topics columns
    relevant_videos = data[data['keyphrases'].apply(lambda x: query in x) | data['topics'].apply(lambda x: query in x)]

    print(relevant_videos['video_url'])
    # Get a list of relevant video URLs
    video_urls = relevant_videos['video_url'].tolist()
    
    print(video_urls)

    # Render the search results template with the relevant video names and search query
    return render_template('search_results.html', query=query, video_urls=relevant_videos['video_url'].tolist())

@app.route('/play', methods=['POST'])
def play():
    # Get the selected video URL from the form
    video_url = request.form['video_url']
    
    # Open the video URL in a web browser
    webbrowser.open(video_url)
    
    # Render the video player template
    return render_template('play.html', video_url=video_url)

if __name__ == '__main__':
    app.run(debug=True)
