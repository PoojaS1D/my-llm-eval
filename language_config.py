# language_config.py
# NO LONGER importing Language here to prevent triggering problematic babelnet client
LANGUAGE_CONFIG = {
    'high_resource': {
        'en': {'name': 'English', 'code': 'en'},  # KEY IS NOW 'en' (string)
        'es': {'name': 'Spanish', 'code': 'es'},  # KEY IS NOW 'es' (string)
        'fr': {'name': 'French', 'code': 'fr'},
        'de': {'name': 'German', 'code': 'de'},
        'it': {'name': 'Italian', 'code': 'it'},
        'pt': {'name': 'Portuguese', 'code': 'pt'},
        'ru': {'name': 'Russian', 'code': 'ru'},
        'zh': {'name': 'Chinese', 'code': 'zh'},
        'ja': {'name': 'Japanese', 'code': 'ja'},
        'ko': {'name': 'Korean', 'code': 'ko'},
        'ar': {'name': 'Arabic', 'code': 'ar'},
        'tr': {'name': 'Turkish', 'code': 'tr'},
        'nl': {'name': 'Dutch', 'code': 'nl'},
        'pl': {'name': 'Polish', 'code': 'pl'},
        'sv': {'name': 'Swedish', 'code': 'sv'},
        'no': {'name': 'Norwegian', 'code': 'no'},
        'da': {'name': 'Danish', 'code': 'da'},
        'fi': {'name': 'Finnish', 'code': 'fi'},
        'cs': {'name': 'Czech', 'code': 'cs'},
        'ro': {'name': 'Romanian', 'code': 'ro'},
        'hu': {'name': 'Hungarian', 'code': 'hu'},
        'uk': {'name': 'Ukrainian', 'code': 'uk'},
        'he': {'name': 'Hebrew', 'code': 'he'},
        'bg': {'name': 'Bulgarian', 'code': 'bg'},
        'el': {'name': 'Greek', 'code': 'el'}
    },

    'medium_resource': {
        'hr': {'name': 'Croatian', 'code': 'hr'},
        'sr': {'name': 'Serbian', 'code': 'sr'},
        'sk': {'name': 'Slovak', 'code': 'sk'},
        'sl': {'name': 'Slovenian', 'code': 'sl'},
        'lt': {'name': 'Lithuanian', 'code': 'lt'},
        'lv': {'name': 'Latvian', 'code': 'lv'},
        'et': {'name': 'Estonian', 'code': 'et'},
        'th': {'name': 'Thai', 'code': 'th'},
        'vi': {'name': 'Vietnamese', 'code': 'vi'},
        'ms': {'name': 'Malay', 'code': 'ms'},
        'fa': {'name': 'Persian', 'code': 'fa'},
        'id': {'name': 'Indonesian', 'code': 'id'},
        'ta': {'name': 'Tamil', 'code': 'ta'},
        'hi': {'name': 'Hindi', 'code': 'hi'},
        'bn': {'name': 'Bengali', 'code': 'bn'}
    },

    'low_resource': {
        'sw': {'name': 'Swahili', 'code': 'sw'},
        'is': {'name': 'Icelandic', 'code': 'is'},
        'mt': {'name': 'Maltese', 'code': 'mt'},
        'ga': {'name': 'Irish', 'code': 'ga'},
        'cy': {'name': 'Welsh', 'code': 'cy'},
        'bs': {'name': 'Bosnian', 'code': 'bs'},
        'ka': {'name': 'Georgian', 'code': 'ka'},
        'am': {'name': 'Amharic', 'code': 'am'},
        'uz': {'name': 'Uzbek', 'code': 'uz'},
        'tl': {'name': 'Tagalog', 'code': 'tl'}
    }
}
