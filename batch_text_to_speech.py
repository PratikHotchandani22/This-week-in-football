from melo.api import TTS
import os
from pydub import AudioSegment

def split_text(text, max_length=200):
    # Split text into chunks of max_length characters, ensuring we don't cut words in half
    words = text.split()
    chunks = []
    current_chunk = ""
    
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += " " + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def merge_audio(files, output_path):
    # Merge multiple audio files into one
    combined = AudioSegment.empty()
    for file in files:
        audio = AudioSegment.from_wav(file)
        combined += audio
    
    combined.export(output_path, format="wav")

def main():
    speed = 0.92
    device = 'cpu'  # Automatically uses GPU if available

    # Input text
    text = """
    Welcome to This Week in Football, I’m Pratik, and I’m here to bring you the biggest stories from around the world of football. 
    From dramatic last-minute goals to shocking upsets, it’s been an action-packed week, so let’s dive right in! 

    In the English Premier League, Manchester City maintained their lead at the top of the table, but not without a scare. 
    They secured a hard-fought 2-1 victory over Brighton, with Erling Haaland once again proving unstoppable, scoring a brilliant brace. 
    Meanwhile, Arsenal stunned Chelsea at Stamford Bridge with a thrilling 3-2 comeback, with Bukayo Saka’s late winner igniting the Gunners’ title hopes. 

    Over in Spain, Real Madrid extended their unbeaten run with a 1-0 win against Sevilla. Jude Bellingham continued his fine form, scoring his 8th goal of the season. 
    Barcelona, however, stumbled with a 1-1 draw against Athletic Bilbao. Despite the setback, Barça remains just a point behind the league leaders. 

    In Italy, AC Milan suffered a shock 2-0 defeat to Juventus, who are slowly clawing their way back into the title race. 
    Milan’s early red card proved costly, and Juventus capitalized with goals from Federico Chiesa and Dusan Vlahovic. 
    Elsewhere, Napoli got back to winning ways, comfortably beating Roma 3-1, keeping their top-four hopes alive. 

    On the European stage, the Champions League group stage delivered more excitement. 
    Paris Saint-Germain crushed AC Milan 4-1 in a must-win match for both sides. Kylian Mbappé was unstoppable, scoring twice and reminding everyone why he’s one of the best in the world. 
    Bayern Munich also secured their place in the knockout stages with a dominant 3-0 win over Manchester United, continuing their European dominance. 

    In transfer news, rumors are swirling about Liverpool’s interest in French midfielder Aurélien Tchouaméni. 
    The Reds are reportedly lining up a massive offer to bring him to Anfield in the January window. 
    Meanwhile, Chelsea’s search for a new striker continues, with talks ongoing for Brentford’s Ivan Toney, who could make the move in January after his suspension ends. 

    And now for the Goal of the Week. This one goes to Newcastle’s Miguel Almirón, who scored an incredible solo goal against Tottenham, 
    dribbling past three defenders before firing into the top corner. A moment of pure brilliance! 

    That’s all for this week in football! Join me next time for more updates from the world’s biggest leagues and competitions. 
    Until then, keep enjoying the beautiful game, and as always... Let’s kick off!
    """


    model = TTS(language='EN', device=device)
    speaker_ids = model.hps.data.spk2id
    output_path = 'final_output.wav'  # Final merged output file
    temp_files = []  # Store temporary audio files for each chunk
    
    # Split text into chunks of 200 characters
    chunks = split_text(text, max_length=190)
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        temp_output_path = f'temp_chunk_{i}.wav'
        model.tts_to_file(chunk, speaker_ids['EN_INDIA'], temp_output_path, speed=speed)
        temp_files.append(temp_output_path)
    
    # Merge the temporary audio files into one
    merge_audio(temp_files, output_path)
    
    # Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)

main()