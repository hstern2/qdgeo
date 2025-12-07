#ifndef OUT_H
#define OUT_H

#include <cstdio>
#include <iostream>
#include <cassert>
#include "util.h"
using namespace std;

#define classIO(T) \
  friend istream & operator>>(istream &, T &); \
  int read_field(istream &, const char *); \
  void write_field_names() const; \
  friend ostream & operator<<(ostream &, const T &); \
  void write_fields(ostream &) const; \
  void show_self();

#define enumIO(T) \
  istream & operator>>(istream &, T &); \
  ostream & operator<<(ostream &, const T &)

#define SHOW(x) Out() << #x << ": " << x << "\n"

void OutInit();
void OutFinalize();
void OutSetToFile(const char *);
void OutPrintf(const char *, ...);
void SetPrecision(int);
void AddToSearchPath(const char *);

FILE *FileSearch(const char *);
istream *StreamSearch(const char *);

ostream &Out();
ostream *FileStream(const char *);
ostream *AppendFileStream(const char *);

class Str;
void TempName(Str &);
void IndentPush();
void IndentPop();
const char *Indent();

class FatalException { };

#define insist(x) assert(x)

#endif /* OUT_H */
