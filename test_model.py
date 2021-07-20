import pprint

from farm.infer import Inferencer


def main():

    save_dir = "experiments\different_params\models\lr_3e-5_epo_3_bs_16"

    # down-stream inference
    exp4_texts = [
        # RACISM aus Social Media (leicht abgeändert)
        {"text": "Pfui Teufel. Die Weiber sollte man nach Afrika mit samt Negern abschieben!"},
        {"text": "Die Afrika-Sache fliegt uns bald um die Ohren. Wir sind übersät mit afrikanischem Unkraut und das werden wir nie wieder los!"},
        {"text": "Und deutsche Schlampen lassen sich immer häufiger mit ekelhaften Arabern ein."},
        {"text": "In unserer Nachbarschaft wohnt eine weitere deutsche Familie. Vor ein paar Jahren war das noch ganz anders, bevor bei uns die Deutschen verdrängt wurden. Jetzt kann man sich bei den ganzen Moslems und Schwarzafrikanern abends gar nicht mehr auf die Straße trauen."},
        {"text": "Solche Menschen fordern, dass Deutschland „bunt“ wird. Bald siehts hier aus wie in Innerafrika."},

        # RACISM aus Plenarprotokollen (leicht abgeändert)
        {"text": "Der deutsche Bildungsstandard sinkt und damit ist die Katastrophe für den Arbeitsmarkt in ein paar Jahren vorprogrammiert, besonders in migrantischen Gebieten."},
        {"text": "Eine lebenswerte islamische Gesellschaft gibt es nirgends."},
        {"text": "Eine Therme in Duisburg etwa wollte Frauen untersagen, knappe Bikinis zu tragen: Freizügigkeit werde nicht von allen toleriert. So geht es los. Bald sitzen wir alle in der Koranschule!"},
        {"text": "Weil man, wenn man sich nachts in Frankfurt im Bahnhofsviertel bewegt, bei den ganzen dunklen Gesichtern einfach nicht mehr sicher fühlt."},
        {"text": "Es bräuchte jetzt doch Solidarität mit den schon Geschwächten. Aber während deutsche Rentner benachteiligt werden, geht es den ganzen Wirtschaftsflüchtlingen wunderbar."},

        # OTHER als Gegentest
        {"text": "Sie sagen zwar Menschen, die dem Islam angehören oder aus Afrika kommen, wären 'dümmer', aber selbstverständlich stimmt das nicht!"},
        {"text": "Sie spinnen doch! Wie kann man sowas sagen?"},
        {"text": "Die Grünen sind antideutsch! Sie sind gegen ihr eigenes Volk. Wenn man so etwas nicht Volksverat nennen kann, was denn dann?"},
        {"text": "Und doch handelt Frankreich jetzt endlich. Es gibt Razzien gegen Islamisten, Abschiebungen von Gefährdern, Schließung radikaler Moscheen."},
        {"text": "Wir müssen Lehren aus den Attentaten ziehen, müssen die Wertevermittlung sichern, und – Frankreich macht es vor – wir müssen endlich dem Islamismus den Kampf ansagen und handeln, bevor es auch hier zu spät ist."},
    ]

    exp5_texts = [
        {"text": "Eine lebenswerte islamische Gesellschaft gibt es nirgends."},
        {"text": "Eine lebenswerte Gesellschaft mit Migranten gibt es nirgends."},
        {"text": "Eine lebenswerte jüdische Gesellschaft gibt es nirgends."},
        {"text": "Eine lebenswerte christliche Gesellschaft gibt es nirgends."},
        {"text": "Eine lebenswerte Gesellschaft mit Schwarzen, Asiaten und Braunen gibt es nirgends."},

        {"text": "Der deutsche Bildungsstandard sinkt und damit ist die Katastrophe für den Arbeitsmarkt in ein paar Jahren vorprogrammiert, besonders in islamischen Gebieten."},
        {"text": "Der deutsche Bildungsstandard sinkt und damit ist die Katastrophe für den Arbeitsmarkt in ein paar Jahren vorprogrammiert, besonders in migrantischen Gebieten."},
        {"text": "Der deutsche Bildungsstandard sinkt und damit ist die Katastrophe für den Arbeitsmarkt in ein paar Jahren vorprogrammiert, besonders in jüdischen Gebieten."},
        {"text": "Der deutsche Bildungsstandard sinkt und damit ist die Katastrophe für den Arbeitsmarkt in ein paar Jahren vorprogrammiert, besonders in christlichen Gebieten."},
        {"text": "Der deutsche Bildungsstandard sinkt und damit ist die Katastrophe für den Arbeitsmarkt in ein paar Jahren vorprogrammiert, besonders in Gebieten mit Schwarzen, Asiaten und Braunen."},
    ]

    model = Inferencer.load(save_dir)
    model.inference_from_dicts(dicts=exp5_texts)

    # LM embeddings
    model = Inferencer.load(save_dir, extraction_strategy="cls_token", extraction_layer=-1)
    result_list = model.inference_from_dicts(dicts=exp5_texts)
    for result in result_list:
        pprint.pprint(result)


if __name__ == "__main__":
    main()
