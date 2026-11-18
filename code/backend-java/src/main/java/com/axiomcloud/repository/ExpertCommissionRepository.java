package com.axiomcloud.repository;

import com.axiomcloud.model.ExpertCommissionEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ExpertCommissionRepository extends JpaRepository<ExpertCommissionEntity, String> {

    List<ExpertCommissionEntity> findAllByOrderByCreatedAtDesc();

    List<ExpertCommissionEntity> findByModalityOrderByCreatedAtDesc(String modality);

    List<ExpertCommissionEntity> findByStatusOrderByCreatedAtDesc(String status);
}
