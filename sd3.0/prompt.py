def MakeObjectPrompt(object, artist):
    return f"{object}, Art by {artist}"
    # return title + ", Art by " + artist


ObjectArtist = ["Dieter Rams"]

ObjectArtistPrompt = map(MakeObjectPrompt, ObjectArtist)
