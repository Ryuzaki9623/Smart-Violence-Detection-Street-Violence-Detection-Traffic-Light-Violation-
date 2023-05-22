import requests

def get_number_plate(filename, token="483260c5b38609d2a17d8d59e2c5c37dd63c4136"):
    """
    Get detected number plates as a list of (accuracy, string)

    Example:
    [(0.3, "kl20b2020"), (0.7, "kl20b2040")]
    """
    try:
        with open(filename, "rb") as fp:
            response = requests.post(
                "https://api.platerecognizer.com/v1/plate-reader/",
                files={"upload": fp},
                data={"regions": ["in"]},
                headers={"Authorization": "Token " + token},
            )
        response = response.json()
        number_plates = []
        if "results" in response:
            for i in response["results"]:
                for j in i["candidates"]:
                    number_plates.append((j["score"], j["plate"]))

        return number_plates
    except:
        return []


def get_most_accurate_number_plate(
    filename, token="483260c5b38609d2a17d8d59e2c5c37dd63c4136"
):
    """Return the most accurate number plate from the image"""
    try:
        plates = get_number_plate(filename, token)
        # Sort the plates based on accuracy (score)
        plates.sort(reverse=True)
        if plates:
            score, string = plates[0]
            return string
        else:
            return None
    except:
        return None

# Example usage
image_filename = "cropped.png"
most_accurate_plate = get_most_accurate_number_plate(image_filename)
print("Most accurate number plate:", most_accurate_plate)
