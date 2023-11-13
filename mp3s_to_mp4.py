import sys
from moviepy import editor


def create_video_clip(mp3_file):
    """Create a video clip with text from an mp3 file."""
    filename_no_ext = mp3_file.split(".")[0]
    audio_clip = editor.AudioFileClip(mp3_file)
    text_clip = editor.TextClip(
        filename_no_ext, fontsize=64, color="white"
    ).set_duration(audio_clip.duration)
    return editor.CompositeVideoClip(
        [text_clip.set_pos("center")], size=(640, 480)
    ).set_audio(audio_clip)


def main(mp3_files):
    """Main function to create a video from multiple mp3 files."""
    video_clips = [create_video_clip(mp3_file) for mp3_file in mp3_files]
    final_clip = editor.concatenate_videoclips(video_clips)
    final_clip.write_videofile("out.mp4", fps=24, audio_codec="aac")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print("Please provide MP3 file paths as command line arguments.")
