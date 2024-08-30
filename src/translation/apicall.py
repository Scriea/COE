import requests
import logging
import json


class TranslateModule:
    def __init__(self, defaultlang=1):
        self.defaultlang = defaultlang

    def translate(self, query, src_lang, dst_lang):
        langs = {1: "eng_Latn", 2: "hin_Deva", 3: "tam_Taml"}
        url = f"http://127.0.0.1:5000/udaan_project_layout/translate/{langs[src_lang]}/{langs[dst_lang]}"
        # url = f"http://10.10.13.2:5000/udaan_project_layout/translate/{langs[src_lang]}/{langs[dst_lang]}"
        # url = f"http://103.42.51.129:5000/udaan_project_layout/translate/{langs[src_lang]}/{langs[dst_lang]}"

        # Data to be translated
        payload = {"sentence": query}

        # Headers (add any required headers like API key if necessary)
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
            # "Authorization": "Bearer YOUR_API_KEY"  # Uncomment if API key is needed
        }
        if langs[src_lang] == "eng_Latn" and langs[dst_lang] == "hin_Deva":
            # url = "http://103.42.51.129:5001/udaan_project_layout/translate/en/hi/med,med_comp/0"
            url = "http://10.10.13.2:5001/udaan_project_layout/translate/en/hi/med,med_comp/0"
        # url = "http://127.0.0.1:5001/udaan_project_layout/translate/en/hi/med,med_comp/0"
        try:
            # Make the POST request
            response = requests.post(url, data=payload, headers=headers)
            # print(response.text)
            response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
            translated_data = response.json()  # Parse the JSON response
            print(response.json())
            translated_text = translated_data.get(
                "translation", "Translation key not found"
            )
            print(translated_text)
            # Ensure the key exists in the response
            # if "translated_text" in translated_data:
            #     print("Translated Text:", translated_data["translation"])
            # else:
            #     raise ValueError(
            #         "The expected key 'translation' was not found in the response."
            #     )

        except requests.exceptions.ConnectionError as ce:
            logging.error(f"Connection Error: {ce}")
            raise
        except requests.exceptions.HTTPError as he:
            logging.error(f"HTTP Error: {he}")
            raise
        except requests.exceptions.Timeout as te:
            logging.error(f"Timeout Error: {te}")
            raise
        except requests.exceptions.RequestException as re:
            logging.error(f"Request Exception: {re}")
            raise
        # except ValueError as ve:
        #     logging.error(f"Value Error: {ve}")
        #     raise
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

        return translated_text


if __name__ == "__main__":
    trans = TranslateModule()
    try:
        trans.translate(
            "நான் ஒரு மரம் மற்றும் எத்தனால் என் குடலுக்கு நல்லதல்ல. கல்லீரல் மோசமாக உள்ளது.", 3, 1
        )
    except Exception as e:
        print(f"An error occurred during translation: {e}")
