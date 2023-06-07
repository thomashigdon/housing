from collections import namedtuple

# Define a named tuple
Location = namedtuple('Location', ['name', 'city'])
def strfunc(self):
    return f"{self.name}"
Location.__str__ = strfunc
City = namedtuple('City', ['name', 'tax_rate', 'exemption'])

cities = {'cambridge': City('cambridge', 5.86, 2759),
          'somerville': City('somerville', 10.74, 3456),
          'boston': City('boston', 10.19, 3910),
          'brookline': City('brookline', 9.97, 3089),
}

zip_codes_of_interest = {
 '02130': Location('jamaica plain, boston', cities['boston']),
 '02143': Location('cambridge-somerville line', cities['somerville']),
 '02145': Location('magoun square, somerville', cities['somerville']),
 '02141': Location('east cambridge', cities['cambridge']),
 '02142': Location('kendall, cambridge', cities['cambridge']),
 '02139': Location('cambridgeport/central', cities['cambridge']),
 '02138': Location('harvard/west cambridge', cities['cambridge']),
 '02140': Location('north cambridge', cities['cambridge']),
 '02144': Location('davis, somerville', cities['somerville']),
 '02129': Location('charlestown, boston', cities['boston']),
 '02114': Location('beacon hill north slope/west end, boston', cities['boston']),
 '02113': Location('north end, boston', cities['boston']),
 '02109': Location('north end waterfront, boston', cities['boston']),
 '02110': Location('waterfront, boston', cities['boston']),
 '02108': Location('beacon hill/state house, boston', cities['boston']),
 '02111': Location('leather district, boston', cities['boston']),
 '02110': Location('south waterfront, boston', cities['boston']),
 '02210': Location('seaport, boston', cities['boston']),
 '02127': Location('southie, boston', cities['boston']),
 '02118': Location('south end, boston', cities['boston']),
 '02116': Location('back bay, boston', cities['boston']),
 '02119': Location('pru, boston', cities['boston']),
 '02115': Location('huntington/northeastern, boston', cities['boston']),
 '02120': Location('mission hill, boston', cities['boston']),
 '02215': Location('fenway/kenmore, boston', cities['boston']),
 '02446': Location('brookline (north)', cities['brookline']),
 '02445': Location('brookline (south)', cities['brookline']),
 '02134': Location('allston, boston', cities['boston']),
 '02125': Location('north dorchester, boston', cities['boston']),
 '02119': Location('roxbury, boston', cities['boston']),
#'02126': Location('mattapan, boston', cities['boston']),
 '02135': Location('brighton, boston', cities['boston']),
}

interest_rate = 0.0665
down_payment = .25
