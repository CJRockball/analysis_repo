from pathlib import Path

import jwt
import requests
from cryptography.x509 import load_pem_x509_certificate

# public_key_text = (Path(__file__).parent / "../../public_key.pem").read_text()
# public_key = load_pem_x509_certificate(public_key_text.encode()).public_key()

# def decode_and_validate_token(access_token):
#     """
#     Validate ab access token. If the token is valid, it returns the token payload. 
#     """
#     return jwt.decode(access_token, key=public_key, algorithms=['RS256'], audience=['http://127.0.0.1:8000'])

X509_CERT_TEMPLATE = (
    "-----BEGIN CERTIFICATE-----\n{key}\n-----END CERTIFICATE-----"
)

public_keys = requests.get('https://idaas-service-provider.au.auth0.com/.well-known/jwks.json').json()['keys']

def _get_certificate_for_kid(kid):
    """
    Return the public key whose ID matches the provided kid.
    If no match is found, raise an exception.
    """
    for key in public_keys:
        if key['kid'] == kid:
            return key['x5c'][0]
    raise Exception(f'Not matching key found for kid {kid}')


def load_public_key_from_x509_cert(certificate):
    """
    """
    return load_pem_x509_certificate(certificate).public_key

def decode_and_validate_token(access_token):
    """
    """
    unverified_headers = jwt.get_unverified_header(access_token)
    x509_certificate = _get_certificate_for_kid(unverified_headers['kid'])

    public_key = load_public_key_from_x509_cert(
        X509_CERT_TEMPLATE.format(key=x509_certificate).encode("utf-8"))
    
    return jwt.decode(
        access_token,
        key=public_key,
        algorithms=unverified_headers['alg'],
        audience = ['http://127.0.0.1:8000',
                    'https://idaas-service-provider.au.auth0.com/userinfo']
        )

