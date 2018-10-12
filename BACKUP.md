spatial_file = '/data/yfcc100m/worldcitiespop.txt'
  with open(spatial_file, encoding='iso-8859-1') as fin:
    line = fin.readline()
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split(',')
      country_code, city = fields[0], fields[1]
      if ' ' in city:
        continue
      spatial_noun.add(country_code)
      spatial_noun.add(city)