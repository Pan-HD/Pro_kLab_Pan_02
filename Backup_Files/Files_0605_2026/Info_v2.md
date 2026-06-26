- 検出対象

  - クラック

- 応用手法

  - GP ONLY

- 目的

  - GPはクラックの小規模データセットから検出フォローを自動的に構築し，さらに未知のクラック画像に適用できることを検証する．

- 訓練セット・テストセットの分割

  - Training

    - 10

    - ```
      train/
          oriImg_01.png
          tarImg_01.png
      
          ...
      
          oriImg_10.png
          tarImg_10.png
      ```

  - Test

    - 100

    - ```
      test/
          oriImg_001.png
          tarImg_001.png
      
          ...
      
          oriImg_100.png
          tarImg_100.png
      ```

- 