      for batch_t in range(num_batch_t):
        image_np_t, label_dat = sess.run([image_bt_t, label_bt_t])
        # print(image_np_t.shape, label_dat.shape)
        label_dat = np.squeeze(label_dat)
        feed_dict = {gen_t.image_ph:image_np_t}
        label_gen, = sess.run([gen_t.labels], feed_dict=feed_dict)
        # print(label_gen.shape)
        # print(np.argsort(-label_dat)[:10])
        # print(np.argsort(-label_gen)[:10])

        sample_np_t, label_np_t = generate_label(label_dat, label_gen)
        # for sample, label in zip(sample_np_t, label_np_t):
        #   print(sample, label)

        feed_dict = {
          dis_t.image_ph:image_np_t,
          dis_t.sample_ph:sample_np_t,
          dis_t.label_ph:label_np_t,
        }
        _, summary = sess.run([dis_t.train_op, dis_t.summary_op], feed_dict=feed_dict)
        writer.add_summary(summary, batch_t)

        if (batch_t + 1) % 100 != 0:
            continue
        print('#%d' % (batch_t))
