from sense_emu import SenseHat

sense = SenseHat()
sense.clear()

while True:
    temp = sense.temp

    # Console
    print(f"Current temperature: {round(temp,1)}°C")

    # Matrix LED
    sense.show_message(text_string=f"{round(temp)}C", scroll_speed=0.09)
