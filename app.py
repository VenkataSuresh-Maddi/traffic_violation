print("\n🚦 Helmet Detection Traffic Violation System 🚦")
print("1. Image Detection")
print("2. Video Detection")
print("3. Webcam Detection")

choice = input("Enter choice: ")

if choice == "1":
    import inference.image_infer
elif choice == "2":
    import inference.video_infer
elif choice == "3":
    import inference.webcam_infer
else:
    print("❌ Invalid choice")
