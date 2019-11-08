 Data 폴더
  미리 만들어놓은 512x 512 크기의 데이터를 만들어 놓았습니다.
  데이터는 Mantaflow를 활용하여 만든 데이터 입니다.(.uni 파일)

 model 폴더
  TempoGAN에서 제공하는 Model이 들어있습니다.
  https://github.com/thunil/tempoGAN 여기서 확인할 수 있습니다.

 result 폴더
  Super-resolution을 진행하고 난 결과가 저장됩니다.
  time_check 폴더에는 frame당 소요된 시간이 저장되어 있습니다.

 tools 폴더
  여기에는 TempoGAN Model에 관련된 코드들과 Quad_Tree_GPU 코드가 들어있습니다. 

사용법)

main.py를 실행하셔서 

기본 Parameter Setting 원하는대로 하신 뒤
Set.Run_QuadTreeSR() 함수에 분할하고 싶은 Patch size를 입력하면 됩니다.

ex) 512 x 512 데이터를 64 Patch size로 실행하면 가로 Patch 8개 세로 8개 총 64개의 Patch로 나눠 
    Quad Tree를 만들고 Super-resolution을 진행합니다.
