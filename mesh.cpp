#include <iostream>

static int sizes[] = {1, 8, 10}; /* 次数(0はじまり)ごとの分割数 */

/**
 * 指定したポイントが属するメッシュコードを計算します。
 * @param lat 緯度 (南北方向) 単位は十進の度
 * @param lng 経度 (東西方向) 単位は十進の度
 * @param dimension メッシュ次数。1次メッシュなら1、3次メッシュなら3とします。
 * @param meshcode メッシュコードを格納する整数配列の先頭へのポインタ。事前に要素数 2*dimensionを確保します。
 */
void meshcode(double lat, double lng, int dimension, int *mesh) {
  int d; /* 次数カウンタ */
  double y, x; /* lat, lng からできた値を格納する */
  /* 1次メッシュを算出しておく */
  y = lat * 1.5;
  x = lng - 100.0;
  for( d = 0; d < dimension; d++ ) {
    /*
     * (d+1)次のメッシュコードを得る
     * (d+1-1)次のメッシュコードの端数[0,1)が y, xに入っているので、
     * (d+1)次のメッシュコードを得るには分割数に応じた掛け算を行って、
     * 整数部を得る。
     */
    y = y * (double)sizes[d];
    x = x * (double)sizes[d];
    mesh[2*d] = (int)y;
    mesh[2*d+1] = (int)x;
    /*
     * y, x を (d+1)次メッシュコードの端数に更新する
     */
    y = y - (double)mesh[2*d];
    x = x - (double)mesh[2*d+1];
  }
}

int main(void) {
  static int mesh[6]; /* 要素数は 2*次数 */
  double lat = 34.5207434;
  double lng = 133.4235147;
  int d; /* 表示に使うカウンタ */

  meshcode(lat, lng, 3, mesh);

  for( d = 0; d < 3; d++ ) {
    printf("%d %d\n", mesh[2*d], mesh[2*d+1]);
  }

  return 0;
}
