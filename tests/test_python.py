"""Test for python fonctions."""

from animalguessinggame.public.classif_animals10 import concat, classifie_animals10
from animalguessinggame.public.levenstein import distance_levenstein
import animalguessinggame.public.classif_animals10

def test_concat():
    """Test concat with 5236."""
    r = concat([5, 2, 3, 6])
    assert r == "cinq mille deux cent trente six"

def test_distance_levenstein_same_strings():
    """Test levenstein = 0."""
    result = distance_levenstein("hello", "hello")
    assert result == 0

def test_distance_levenstein_one_edit():
    """Test levenstein = 1."""
    result = distance_levenstein("kitten", "sitten")
    assert result == 1

def test_distance_levenstein_different_lengths():
    """Test levenstein = 3."""
    result = distance_levenstein("kitten", "sitting")
    assert result == 3

def test_distance_levenstein_empty_strings():
    """Test levenstein = 4."""
    result = distance_levenstein("", "test")
    assert result == 4

def test_classifie_animals10_good_animal():
    """Test classifieur animals10."""
    result = classifie_animals10("/images_animals10/OIP-YE9OFJVKN8oTj_oDLcV2-wHaFj.jpeg")
    assert result == ["poule", "chicken"]

def test_classifie_animals10_false_animal():
    """Test classifieur animals10."""
    result = classifie_animals10("/images_animals10/OIP-YE9OFJVKN8oTj_oDLcV2-wHaFj.jpeg")
    assert result != ["chat", "cat"]
