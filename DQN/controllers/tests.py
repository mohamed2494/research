from django.test import TestCase
import json
from pprint import pprint
from django.http import HttpResponse
# Create your tests here.


def test(request):
    pprint(11)
    response = {}
    response["key"] = "value"
    return HttpResponse(json.dumps(response),content_type="application/json")