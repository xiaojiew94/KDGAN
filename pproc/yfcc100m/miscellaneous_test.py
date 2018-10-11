
def main():
  print('miscellaneous test')

  file = '/home/xiaojie/Projects/data/yfcc100m/yfcc_rnd/yfcc_rnd.data'
  with open(file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      print(line)
      input()

if __name__ == '__main__':
  main()