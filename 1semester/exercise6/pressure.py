from sense_emu import SenseHat

sense = SenseHat()
sense.clear()


while True:
    pressure = sense.pressure

    # Console
    print(f"Current pressure: {round(pressure,1)}mbar")

    # Matrix LED
    sense.show_message(text_string=f"{round(pressure)}mbar", scroll_speed=0.09)
