"""
Purpose: Flask microservice for predicting handwritten digits.

"""

from flask import Flask, request, jsonify
from flask_restplus import Api, Resource

